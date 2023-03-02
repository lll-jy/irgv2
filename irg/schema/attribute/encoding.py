"""Handler for encoding data."""
from abc import ABC
import os
import pickle
from typing import Optional, Dict, List, Union, Tuple, Literal, Any

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch import nn
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import euclidean_distances
from nltk import word_tokenize, wordpunct_tokenize, casual_tokenize

from .base import BaseAttribute, BaseTransformer
from ...utils.io import load_from, pd_to_pickle
from ...utils.torch import SequenceEmbedding


class _TextTransformer(BaseTransformer, ABC):
    def __init__(self, temp_cache: str = '.temp'):
        super().__init__(temp_cache)
        self._vocab: Optional[Dict[str, List[Union[int, float]]]] = None
        self._vocab_dim = -1

    def load_vocab(self, check_dim: bool = True, **kwargs):
        """
        Load vocabulary file.

        **Args**:

        - `check_dim` (`bool`) [default `True`]: Whether to check the validity of loaded data format.
          It must be able to be interpreted as a `dict` from `str` to a vector of numbers.
          And the length of the vectors should be the same for all words.
          Also, the vocab cannot be empty.
          '[UNK]' is reserved for recognizing unseen values. It will not be checked because input with this value
          is likely to express the same meaning.
        - `kwargs`: Arguments for [load_from](../utils/misc#load_from).
        """
        self._vocab = load_from(**kwargs)
        if check_dim:
            self._check_vocab()

    def _check_vocab(self):
        if not isinstance(self._vocab, Dict):
            raise ValueError(f'The vocabulary must be a dict. Got {type(self._vocab)}.')
        dim = -1
        for k, v in self._vocab.items():
            if dim < 0:
                dim = len(v)
            else:
                if dim != len(v):
                    raise ValueError(f'Encoding dimension does not match. Want {dim}, got {len(v)}.')
        if dim < 0:
            raise ValueError('The vocabulary must not be empty.')

    def _calc_dim(self) -> int:
        return self._vocab_dim

    def _calc_fill_nan(self, original: pd.Series) -> str:
        return '[UNK]'

    def _categorical_dimensions(self) -> List[Tuple[int, int]]:
        return [(0, 1)]


class EncodingTransformer(_TextTransformer):
    """Transformer for encoding data, where each value associates with a vector representation."""
    def __init__(self, temp_cache: str = '.temp'):
        super().__init__(temp_cache)
        self._knn: Optional[KNeighborsClassifier] = None
        self._mean_enc: Optional[List[float]] = None

    def _unload_additional_info(self):
        self._vocab, self._knn, self._mean_enc = None, None, None

    def _load_additional_info(self):
        if os.path.exists(os.path.join(self._temp_cache, 'info.pkl')):
            with open(os.path.join(self._temp_cache, 'info.pkl'), 'rb') as f:
                loaded = pickle.load(f)
            self._vocab, self._knn, self._mean_enc = loaded['vocab'], loaded['knn'], loaded['mean_enc']
        else:
            self._vocab, self._mean_enc = None, None
            self._knn = KNeighborsClassifier(n_neighbors=1)

    def _save_additional_info(self):
        with open(os.path.join(self._temp_cache, 'info.pkl'), 'wb') as f:
            pickle.dump({
                'vocab': self._vocab,
                'knn': self._knn,
                'mean_enc': self._mean_enc
            }, f)

    @property
    def atype(self) -> str:
        return 'encoding'

    def _fit(self, original: pd.Series, nan_info: pd.DataFrame):
        values = [*self._vocab.values()]
        self._mean_enc = np.array(values).mean(axis=0)
        self._vocab_dim = len(values[0])
        self._knn.fit(values, [*self._vocab.keys()])
        transformed = self._transform(nan_info)
        self._transformed_columns = transformed.columns
        pd_to_pickle(transformed, self._transformed_path)

    def _transform(self, nan_info: pd.DataFrame) -> pd.DataFrame:
        nan_info['original'] = nan_info['original'].astype(str)
        col_names = [f'enc_{i}' for i in range(self._vocab_dim)]
        col_names = ['is_nan'] + col_names
        transformed = pd.DataFrame(columns=col_names)
        if self._has_nan:
            transformed['is_nan'] = nan_info['is_nan']
        for i, row in nan_info.iterrows():
            if row['is_nan']:
                transformed.iloc[i, 1:] = 0
            else:
                transformed.iloc[i, 1:] = self._vocab.get(row['original'], self._mean_enc)
        return transformed

    def _inverse_transform(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(self._knn.predict(data))


class EncodingAttribute(BaseAttribute):
    """Attribute for encoding data, where each value associates with a vector representation."""

    def __init__(self, name: str, vocab_file: str, engine: Optional[str] = None, values: Optional[pd.Series] = None,
                 temp_cache: str = '.temp'):
        """
        **Args**:

        - `name` (`str`): Name of the attribute.
        - `vocab_file` (`str`): File to vocabulary.
        - `engine` (`Optional[str]`): Engine for [load_from](../utils/misc#load_from).
        - `values` (`Optional[pd.Series]`): Data of the attribute (that is used for fitting normalization transformers).
        - `temp_cache` (`str`): Directory path to save cached temporary files. Default is `.temp`.
        """
        super().__init__(name, 'encoding', values, temp_cache)
        self._vocab_file, self._engine = vocab_file, engine

    def _create_transformer(self):
        self._transformer = EncodingTransformer(self._temp_cache)
        self._transformer.load_vocab(file_path=self._vocab_file, engine=self._engine)

    def __copy__(self) -> "EncodingAttribute":
        new_attr = super().__copy__()
        new_attr.__class__ = EncodingAttribute
        new_attr._vocab_file, new_attr._engine = self._vocab_file, self._engine
        return new_attr


class ShortTextTransformer(_TextTransformer):
    """Transformer for short text data,
    where each value associates with the sum or mean of the embeddings of each word."""

    def __init__(self, temp_cache: str = '.temp', vocab_version: str = 'gpt100',
                 agg_func: Literal['mean', 'sum'] = 'sum', tolower: bool = True,
                 tokenization: Literal['word', 'wordpunct', 'casual'] = 'casual'):
        """
        **Args**

        - `temp_cache` (`str`): Same for `BaseTransformer`.
        - `vocab_version` (`str`): Vocabulary version. Can be either pretrained ones like `'gpt100'` or
          path to the file holding the embeddings of each word as a `dict`.
        - `agg_func` (`Literal['mean', 'sum']`): Aggregation across different words for the short text description.
          Default is `'sum'`.
        - `tolower` (`bool`): Whether to make the letters lower-case.
        - `tokenization` (`Literal['word', 'wordpunct', 'casual']`): Tokenization function in NLTK.
          Default is `'casual'`.
        """
        super().__init__(temp_cache)
        self._vocab_path, self._vocab_size = self._get_vocab_info(vocab_version)
        self._acc_func = agg_func
        self._vocab = None
        self._emb_map = None
        self._categories = None
        self._tolower = tolower
        self._tokenization = tokenization

    @staticmethod
    def _get_vocab_info(vocab_version: str) -> (str, int):
        pre_trained = {
            'gpt100': ('', 100)  # TODO: download actual ones
        }
        if vocab_version in pre_trained:
            return pre_trained[vocab_version]
        vocab = load_from(vocab_version)
        vector = next(iter(vocab.values()))
        return vocab_version, len(vector)

    @property
    def atype(self) -> str:
        return 'shorttext'

    def _fit(self, original: pd.Series, nan_info: pd.DataFrame):
        unique_values = original.unique()
        self._emb_map = np.zeros((len(unique_values), self._vocab_size))
        self._categories = []
        for i, value in enumerate(unique_values):
            if pd.isnull(value):
                enc = np.ndarray(self._vocab['[UNK]'])
                value = ['[UNK]']
            else:
                words = self._tokenize(value)
                enc = np.zeros(self._vocab_size)
                cnt = 0
                for w in words:
                    if w in self._vocab:
                        enc += np.ndarray(self._vocab[w])
                        cnt += 1
                if self._acc_func == 'mean':
                    enc = enc / cnt
            self._emb_map[i] = enc
            self._categories.append(value)

    def _tokenize(self, sent: str) -> List[str]:
        if self._tokenization == 'word':
            tokens = word_tokenize(sent)
        elif self._tokenization == 'wordpunct':
            tokens = wordpunct_tokenize(sent)
        elif self._tokenization == 'casual':
            tokens = casual_tokenize(sent)
        else:
            raise NotImplementedError(f'Tokenization of {self._tokenization} is not recognized.')
        if self._tolower:
            tokens = [t.tolower() for t in tokens]
        return tokens

    def _unload_additional_info(self):
        self._vocab = None
        self._emb_map = None
        self._categories = None

    def _load_additional_info(self):
        self.load_vocab(file_path=self._vocab_path)
        if os.path.exists(os.path.join(self._temp_cache, 'info.pkl')):
            with open(os.path.join(self._temp_cache, 'info.pkl'), 'rb') as f:
                loaded = pickle.load(f)
            self._emb_map, self._categories = loaded['emb_map'], loaded['categories']
        else:
            self._emb_map = np.zeros((0, self._vocab_size))
            self._categories = []

    def _save_additional_info(self):
        with open(os.path.join(self._temp_cache, 'info.pkl'), 'wb') as f:
            pickle.dump({
                'emb_map': self._emb_map,
                'categories': self._categories
            }, f)

    def _transform(self, nan_info: pd.DataFrame) -> pd.DataFrame:
        nan_info['original'] = nan_info['original'].astype(str)
        col_names = [f'enc_{i}' for i in range(self._vocab_size)]
        col_names = ['is_nan'] + col_names
        transformed = pd.DataFrame(columns=col_names)
        if self._has_nan:
            transformed['is_nan'] = nan_info['is_nan']
        for i, row in nan_info.iterrows():
            if row['is_nan']:
                transformed.iloc[i, 1:] = 0
            else:
                try:
                    cat_id = self._categories.index(row['original'])
                    transformed.iloc[i, 1:] = self._emb_map[cat_id]
                except ValueError:
                    assert isinstance(self._emb_map, np.ndarray)
                    transformed.iloc[1, 1:] = self._emb_map.mean(axis=0)
        return transformed

    def _inverse_transform(self, data: pd.DataFrame) -> pd.Series:
        normalized = data.iloc[:, 1:].values
        distances = euclidean_distances(normalized, self._emb_map)
        result = []
        for row in distances:
            norm_row = torch.tensor(-row).softmax(dim=-1).numpy()
            norm_row = norm_row / norm_row.sum()
            choice = np.random.choice(
                a=range(len(self._emb_map)),
                p=norm_row
            )
            result.append(self._categories[choice])
        return pd.Series(result)


class ShortTextAttribute(BaseAttribute):
    """Attribute for short text data,
    where each value associates with the sum or mean of the embeddings of each word."""

    def __init__(self, name: str, values: Optional[pd.Series] = None, temp_cache: str = '.temp',
                 **kwargs):
        """
        **Args**:

        - `name` (`str`): Name of the attribute.
        - `values` (`Optional[pd.Series]`): Data of the attribute (that is used for fitting normalization transformers).
        - `kwargs`: Arguments to `ShortTextTransformer`.
        - `temp_cache` (`str`): Directory path to save cached temporary files. Default is `.temp`.
        """
        self._kwargs = kwargs
        super().__init__(name, 'encoding', values, temp_cache)

    def _create_transformer(self):
        self._transformer = ShortTextTransformer(temp_cache=self._temp_cache, **self._kwargs)

    def __copy__(self) -> "ShortTextAttribute":
        new_attr = super().__copy__()
        new_attr.__class__ = ShortTextAttribute
        new_attr._kwargs = self._kwargs
        return new_attr


class EmbeddingTransformer(BaseTransformer):
    """
    Embedding transformer, for series table.
    The tokens are likely to form a sequence like words in a language.
    This transformer will be trained as an CBOW.

    The transformed columns (after `is_nan`) form embedding vectors.
    """
    def __init__(self, temp_cache: str = '.temp', embedding_dim: int = 50, half_window_size: int = 3,
                 hidden_dim: int = 128, lr: float = 0.001, epoch: int = 10, batch_size: int = 64):
        """
        **Args**:

        - `temp_cache` (`str`): Same for `BaseTransformer`.
        - `embedding_dim` (`int`): Embedding dimensions. Default is 50.
        - `half_window_size` (`int`): Half window size for the embedding model trained based on N-Gram. Default is 3.
        - `hidden_dim` (`int`): Hidden dimension for embedding. Default is 128.
        - `lr` (`float`): Learning rate. We will use Adam optimizer. Default is 0.001.
        - `epoch` (`int`): Number of epochs to train. Default is 10.
        - `batch_size (`int`): Batch size during training. Default is 64.
        """
        self._embedding_dim = embedding_dim
        self._half_window_size = half_window_size
        self._hidden_dim = hidden_dim
        self._lr = lr
        self._epoch, self._batch_size = epoch, batch_size
        self._label2id, self._id2label = None, None
        self._bos_id, self._pad_id, self._eos_id = 0, 1, 2
        self._model = None
        super().__init__(temp_cache)

    def _save_additional_info(self):
        with open(os.path.join(self._temp_cache, 'info.pkl'), 'wb') as f:
            pickle.dump({
                'label2id': self._label2id,
                'id2label': self._id2label,
                'model': self._model
            }, f)

    def _load_additional_info(self):
        if os.path.exists(os.path.join(self._temp_cache, 'info.pkl')):
            with open(os.path.join(self._temp_cache, 'info.pkl'), 'rb') as f:
                loaded = pickle.load(f)
            self._label2id, self._id2label = loaded['label2id'], loaded['id2label']
            self._model = loaded['model']
        else:
            self._label2id, self._id2label = {}, {}

    def _unload_additional_info(self):
        self._label2id, self._id2label = None, None
        self._model = None

    @property
    def atype(self) -> str:
        return 'embedding'

    def _calc_dim(self) -> int:
        return self._embedding_dim

    def _calc_fill_nan(self, original: pd.Series) -> Any:
        return 'nan'

    def _fit(self, original: pd.Series, nan_info: pd.DataFrame):
        categories = nan_info['original'].unique()
        for i, cat in enumerate(categories):
            self._label2id[str(cat)] = i + 3
            self._id2label[i + 3] = str(cat)
        self._model = SequenceEmbedding(
            vocab_size=len(categories) + 3,
            embedding_dim=self._embedding_dim,
            context_size=2 * self._half_window_size,
            hidden_dim=self._hidden_dim
        )
        optimizer = Adam(
            self._model.parameters(),
            lr=self._lr
        )

        context = []
        target = []
        pad_num = max(0, self._half_window_size - 1)
        prefix = [self._pad_id] * pad_num + [self._bos_id]
        suffix = [self._eos_id] + [self._pad_id] * pad_num
        for _, group in nan_info['original'].groupby(level=0, dropna=False):
            length = len(group)
            group = prefix + [
                self._label2id[item]
                for item in group.values
            ] + suffix
            print('full group::', length, len(group), pad_num, flush=True)
            for mid in range(pad_num + 1, pad_num + 1 + length):
                print('full range', mid-self._half_window_size, mid+self._half_window_size+1, len(group), flush=True)
                context.append(group[mid-self._half_window_size:mid-1] + group[mid+1:mid+self._half_window_size+1])
                target.append(group[mid])

        context = torch.LongTensor(context)
        target = torch.LongTensor(target)
        print('context and target sizes', context.shape, target.shape, flush=True)
        loss_func = nn.NLLLoss
        for epoch in range(self._epoch):
            iterator = tqdm(range(0, len(context), self._batch_size), desc=f'Training Embedding [{epoch}]')
            total_loss = torch.tensor(0)
            for idx in iterator:
                batch_context = context[idx:idx+self._batch_size]
                batch_target = target[idx:idx+self._batch_size]
                self._model.zero_grad()
                log_probs = self._model(batch_context)
                loss = loss_func(log_probs, batch_target)
                loss.backward()
                optimizer.step()
                total_loss += loss
                iterator.set_description(
                    f'Training Embedding [{epoch}]: loss {loss.item():.3f}, '
                    f'avg loss {total_loss.item()/(idx+batch_target.shape[0]):.3f}'
                )

    def _transform(self, nan_info: pd.DataFrame) -> pd.DataFrame:
        emb_columns = [f'emb_{i}' for i in range(self._embedding_dim)]
        transformed = pd.DataFrame(columns=['is_nan'] + emb_columns)
        transformed['is_nan'] = nan_info['is_nan']
        for i, v in nan_info.iterrows():
            # (WT x W)-1 x (WT x W) x id = (WT x W)-1 x WT x emb
            # id = W-1 x emb
            transformed.loc[i, emb_columns] = self._model.embeddings.weight[self._label2id[str(v)]].flatten()
        return transformed

    def _inverse_transform(self, data: pd.DataFrame) -> pd.Series:
        result = []
        for row in data.iterrows():
            distance = torch.norm(self._model.embeddings.weight.data[3:] - torch.tensor(row).view(1, -1), dim=1)
            result.append(self._id2label[torch.argmin(distance) + 3])
        return pd.Series(result)

    def _categorical_dimensions(self) -> List[Tuple[int, int]]:
        return [(0, 1)]


class EmbeddingAttribute(BaseAttribute):
    """
    Embedding attribute, for series table.
    The tokens are likely to form a sequence like words in a language.
    """
    def __init__(self, name: str, values: Optional[pd.Series] = None, temp_cache: str = '.temp',
                 embedding_dim: int = 50, half_window_size: int = 3,
                 hidden_dim: int = 128, lr: float = 0.001, epoch: int = 10, batch_size: int = 64):
        """
        **Args**:

        - `name` (`str`): Name of the attribute.
        - `values` (`Optional[pd.Series]`): Data of the attribute (that is used for fitting normalization transformers).
        - `temp_cache` (`str`): Directory path to save cached temporary files. Default is `.temp`.
        - `embedding_dim` (`int`): Hidden dimension of the embedding attribute.
        - `half_window_size` (`int`): Half window size for the embedding model trained based on N-Gram. Default is 3.
        - `hidden_dim` (`int`): Hidden dimension for embedding. Default is 128.
        - `lr` (`float`): Learning rate. We will use Adam optimizer. Default is 0.001.
        - `epoch` (`int`): Number of epochs to train. Default is 10.
        - `batch_size (`int`): Batch size during training. Default is 64.
        """
        self._embedding_dim = embedding_dim
        self._half_window_size = half_window_size
        self._hidden_dim = hidden_dim
        self._lr = lr
        self._epoch, self._batch_size = epoch, batch_size
        super().__init__(name, 'embedding', values, temp_cache)

    def _create_transformer(self):
        self._transformer = EmbeddingTransformer(
            temp_cache=self._temp_cache, embedding_dim=self._embedding_dim, half_window_size=self._half_window_size,
            hidden_dim=self._hidden_dim, lr=self._lr, epoch=self._epoch, batch_size=self._batch_size
        )

    def __copy__(self) -> "EmbeddingAttribute":
        new_attr = super().__copy__()
        new_attr.__class__ = EmbeddingAttribute
        return new_attr
