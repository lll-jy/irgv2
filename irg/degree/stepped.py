import os
from typing import List

import pandas as pd
import torch
from torch import Tensor
from torch.nn import Embedding, Sequential, Linear, ReLU, ModuleDict, Dropout, MSELoss, Identity, CrossEntropyLoss
from torch.optim import Adam, Optimizer
from tqdm import tqdm

from .base import DegreeTrainer
from ..schema import Table, Database
from ..utils.misc import make_dataloader, convert_data_as
from ..utils.dist import get_device, to_device
from ..utils.io import load_state_dict, save_state_dict


class DegreeSteppedTrainer(DegreeTrainer):
    """Degree prediction by each foreign key one by one."""
    def __init__(self, embedding_dim: int = 100, lr: float = 0.001, n_epoch: int = 10, batch_size: int = 64,
                 hidden_dim: int = 100, n_layer: int = 3, dropout: float = 0.1, emb_epoch: int = 3, **kwargs):
        """
        **Args**:

        - `embedding_dim` (`int`): The embedding dimension to encode a row in each parent table.
        - `lr` (`float`): Learning rate. Default is 0.001.
        - `n_epoch` (`int`): Number of epochs to train. Default is 10.
        - `batch_size` (`int`): Batch size during training. Default is 64.
        - `hidden_dim` (`int`): Number of hidden dimensions for models. Default is 100.
        - `n_layer` (`int`): Number of layers for models. Default is 3.
        - `dropout` (`float`): Dropout of the models. Default is 0.1.
        - `emb_epoch` (`int`): Number of epochs to train embeddings before the model. Default is 3.
        """
        super().__init__(**kwargs)
        self._embedding_dim = embedding_dim
        self._lr = lr
        self._n_epoch = n_epoch
        self._batch_size = batch_size
        self._hidden_dim = hidden_dim
        self._n_layer = n_layer
        self._dropout = dropout
        self._emb_epoch = emb_epoch
        self._model_size = [(0, 0)] * len(self._foreign_keys)  # (input size, embedding vocab size)
        os.makedirs(os.path.join(self._cache_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(self._cache_dir, 'encoded'), exist_ok=True)
        self._device = get_device()

    def _model_path_for(self, idx: int) -> str:
        return os.path.join(self._cache_dir, 'models', f'fk{idx}_model.pt')

    def _emb_path_for(self, idx: int) -> str:
        return os.path.join(self._cache_dir, 'models', f'fk{idx}_emb.pt')

    def _match_path_for(self, idx: int) -> str:
        return os.path.join(self._cache_dir, 'models', f'fk{idx}_match.pt')

    def _fit(self, data: Table, context: Database):
        # 1. fit degrees of combination thus far
        # 2. fit embeddings for next foreign key
        # 3. fit matching model from combination thus far to inverse embedding
        deg_data = data.data('degree', normalize=False, with_id='this')
        deg_norm = data.data('degree', normalize=True, with_id='none')
        train_cols = []
        fk_cols = []
        for i, fk in enumerate(self._foreign_keys):
            train_cols.extend([(a, b, c) for a, b, c in deg_norm.columns if a.startswith(f'fk{i}:')])
            train_data = deg_norm[train_cols]
            fk_cols.extend(fk.left)
            degrees = deg_data.groupby(fk_cols)[('', 'degree')].sum()
            self._fit_pred_model(train_data, degrees, i)

            if i == len(self._foreign_keys) - 1:
                return
            parent_normalized = context.augmented_till(self._foreign_keys[i+1].parent, data.name, with_id='none')
            self._fit_embedding(parent_normalized, i + 1)
            parent_fk = context.augmented_till(self._foreign_keys[i+1].parent, data.name,
                                               normalized=False, with_id='this')[self._foreign_keys[i+1].right]
            next_table_data = {v: i for i, v in enumerate(deg_data[self._foreign_keys[i + 1].left])}
            indices = parent_fk.apply(lambda x: next_table_data[x])
            self._fit_matching(train_data, indices, i)

    def _make_mlp(self, in_dim: int, out_dim: int, path: str) -> Sequential:
        in_dim = [in_dim] + [self._hidden_dim] * (self._n_layer - 1)
        out_dim = [self._hidden_dim] * (self._n_layer - 1) + [out_dim]
        act = [ReLU() for _ in range(self._n_layer - 1)] + [Identity()]
        model = Sequential(*[
            ModuleDict({
                'layer': Linear(i, o),
                'dropout': Dropout(self._dropout),
                'act': a
            }) for i, o, a in zip(in_dim, out_dim, act)
        ])
        if os.path.exists(path):
            load_state_dict(model, path)
        return model

    def _make_pred_model(self, idx: int) -> Sequential:
        return self._make_mlp(self._model_size[idx][0], 1, self._model_path_for(idx))

    def _fit_pred_model(self, x: pd.DataFrame, y: pd.Series, idx: int):
        self._model_size[idx] = x.shape[-1], self._model_size[idx][1]
        model = self._make_pred_model(idx)
        if os.path.exists(self._model_path_for(idx)):
            return
        optim = Adam(model.parameters(), lr=self._lr)
        criterion = MSELoss()
        y = pd.DataFrame({'degree': y})
        x = convert_data_as(x)
        y = convert_data_as(y)
        model, x, y, model, criterion, optim = to_device((model, x, y, model, criterion, optim), self._device)
        dataloader = make_dataloader(x, y, batch_size=self._batch_size)
        for e in range(self._n_epoch):
            losses = []
            iterator = tqdm(dataloader, desc=f'DegFk{idx} of {self._name} Epoch[{e}]: Loss: 0.000, Avg: 0.000')
            for ctx, deg in iterator:
                out = model(ctx)
                loss = criterion(out, deg)
                losses.append(loss)
                self._wrap_step(optim, loss, iterator, losses, idx, e)
        save_state_dict(model, self._model_path_for(idx))

    def _make_embedding_model(self, idx: int) -> Embedding:
        embeddings = Embedding(
            num_embeddings=self._model_size[idx][1],
            embedding_dim=self._embedding_dim
        )
        if os.path.exists(self._emb_path_for(idx)):
            load_state_dict(embeddings, self._emb_path_for(idx))
        return embeddings

    def _fit_embedding(self, normalized: pd.DataFrame, idx: int):
        self._model_size[idx] = self._model_size[idx][1], normalized.shape[-1]
        embeddings = self._make_embedding_model(idx)
        if os.path.exists(self._emb_path_for(idx)):
            return
        dataloader = make_dataloader(normalized, torch.LongTensor(range(len(normalized))), batch_size=self._batch_size)
        optim = Adam(embeddings.parameters(), lr=self._lr)
        embeddings, normalized, optim = to_device((embeddings, normalized, optim), self._device)

        for e in range(self._emb_epoch):
            iterator = tqdm(dataloader, desc=f'Matching fk{idx} in {self._name} Epoch [{e}]: Loss: 0.000, Avg: 0.000')
            losses = []
            for embed, tgt in iterator:
                weight = embeddings.weight.clone() @ embed.t()
                encoded = embeddings(tgt) @ embed.t()
                out = (weight.unsqueeze(0) + encoded.unsqueeze(1))
                loss = out.sigmoid().prod()
                losses.append(loss)

                self._wrap_step(optim, loss, iterator, losses, idx, e)
        save_state_dict(embeddings, self._emb_path_for(idx))

    def _make_match_model(self, idx: int) -> Sequential:
        return self._make_mlp(self._model_size[idx][0], self._embedding_dim, self._match_path_for(idx))

    def _fit_matching(self, x: pd.DataFrame, target: pd.Series, idx: int):
        embeddings = self._make_embedding_model(idx + 1)
        model = self._make_match_model(idx)
        if os.path.exists(self._model_path_for(idx)):
            return

        optim = Adam(list(model.parameters()) + list(embeddings.parameters()), lr=self._lr)
        criterion = CrossEntropyLoss()
        x = convert_data_as(x, 'torch')
        y = convert_data_as(pd.DataFrame({'match': target}), 'torch')
        model, embeddings, criterion, x, y, optim = to_device(
            (model, embeddings, criterion, x, y, optim), self._device)
        dataloader = make_dataloader(x, y, batch_size=self._batch_size, shuffle=True)

        for e in range(self._n_epoch):
            iterator = tqdm(dataloader,
                            desc=f'Training fk{idx} in {self._name} Epoch [{e}]: Loss: 0.000, Avg: 0.000')
            losses = []
            for ctx, next_idx in iterator:
                out = model(ctx)
                distance = torch.norm(embeddings.weight.data - out, dim=1)
                distance = distance.max() - distance
                loss = criterion(distance, next_idx)
                losses.append(loss)

                self._wrap_step(optim, loss, iterator, losses, idx, e)
        save_state_dict(embeddings, self._emb_path_for(idx))
        save_state_dict(model, self._match_path_for(idx))

    def _wrap_step(self, optim: Optimizer, loss: Tensor, iterator: tqdm, losses: List[Tensor], idx: int, e: int):
        optim.zero_grad()
        loss.backward()
        optim.step()

        iterator.set_description(
            f'Matching fk{idx} in {self._name} Epoch [{e}]: '
            f'Loss: {loss.item():.3f}, Avg: {sum(losses).item() / len(losses):.3f}'
        )
