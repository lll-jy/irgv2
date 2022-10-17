from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from ctgan.data_sampler import DataSampler as CTGANDataSampler
from ctgan.data_transformer import SpanInfo
from sklearn.neighbors import KNeighborsClassifier
import torch
from torch import Tensor

from .base import TabularTrainer
from ..utils import InferenceOutput


class DataSampler(CTGANDataSampler):
    def __init__(self, data: Tensor, context: Tensor, info: List[List[SpanInfo]]):
        super().__init__(data, info, True)

        self._has_context = context.shape[1] > 0
        if self._has_context:
            self._knn_context = []
            st = 0
            current_id = 0
            current_cond_st = 0
            for column_info in info:
                if self._is_discrete_column(column_info):
                    span_info = column_info[0]
                    ed = st + span_info.dim
                    y = data[:, st:ed].argmax(axis=1)
                    knn = KNeighborsClassifier()
                    knn.fit(context, y)
                    self._knn_context.append(knn)
                    current_cond_st += span_info.dim
                    current_id += 1
                    st = ed
                else:
                    st += sum([span_info.dim for span_info in column_info])
        else:
            self._knn_context = None

        delattr(self, '_data')

    @staticmethod
    def _is_discrete_column(column_info: List[SpanInfo]):
        return (len(column_info) == 1
                and column_info[0].activation_fn == 'softmax')

    def sample_condvec(self, unknown: Tensor) -> (Tensor, Tensor, Tensor, Tensor):
        if self._n_discrete_columns == 0:
            return [torch.zeros(unknown.shape[0], 0) for _ in range(4)]
        batch = unknown.shape[0]
        discrete_column_id = torch.from_numpy(np.random.choice(
            np.arange(self._n_discrete_columns), batch))

        mask = np.zeros(batch, self._n_discrete_columns, dtype=torch.float32)
        mask[np.arange(batch), discrete_column_id] = 1
        cond = torch.zeros(batch, self._n_categories, dtype=torch.float32)
        st = self._discrete_column_cond_st[discrete_column_id]
        width = self._discrete_column_n_category[discrete_column_id]
        category_id_in_col = []
        for i, row_st, row_width, unknown_row in zip(range(batch), st, width, unknown):
            cond[i, st:st+width] = unknown_row[st:st+width]
            category_id_in_col.append(unknown_row[st:st+width].argmax(dim=-1))
        category_id_in_col = torch.tensor(category_id_in_col)
        return cond, mask, discrete_column_id, category_id_in_col

    def sample_original_condvec(self, known: Tensor) -> Tensor:
        if self._n_discrete_columns == 0:
            return torch.zeros(*known.shape)
        cond = torch.zeros(known.shape[0], self._n_categories, dtype=torch.float32)
        discrete_column_id = torch.from_numpy(np.random.choice(
            np.arange(self._n_discrete_columns), known.shape[0]))
        for i, row in enumerate(known):
            discrete_col = discrete_column_id[i]
            choice = self._choice_index(discrete_col, row)
            st = self._discrete_column_cond_st[discrete_col]
            width = self._discrete_column_n_category[discrete_col]
            empty = torch.zeros(width)
            empty[choice] = 1
            cond[i, st:st+width] = empty
        return cond

    def _choice_index(self, discrete_column_id: int, known_row: Tensor) -> np.ndarray:
        if not self._has_context:
            return super()._random_choice_prob_index(discrete_column_id)
        knn: KNeighborsClassifier = self._knn_context[discrete_column_id]
        known = torch.stack([known_row])
        pred = knn.predict(known)
        return pred


class CTGANOutput(InferenceOutput):
    def __init__(self, fake: Tensor, discr_out: Optional[Tensor] = None):
        super().__init__(fake)
        self.fake = fake
        """Fake data generated."""
        self.discr_out = discr_out
        """Discriminator output."""


class CTGANTrainer(TabularTrainer):
    def __init__(self, embedding_dim: int = 128,
                 generator_dim: Tuple[int, ...] = (256, 256), discriminator_dim: Tuple[int, ...] = (256, 256),
                 pac: int = 10, discriminator_step: int = 1, **kwargs):
        super().__init__(**{
            n: v for n, v in kwargs.items() if
            n in {'distributed', 'autocast', 'log_dir', 'ckpt_dir', 'descr',
                  'cat_dims', 'known_dim', 'unknown_dim'}
        })

        self._sampler: Optional[DataSampler] = None

    def _load_content_from(self, loaded: Dict[str, Any]):
        pass

    def _construct_content_to_save(self) -> Dict[str, Any]:
        pass

    def __reduce__(self):
        pass

    def _construct_sampler(self, known: Tensor, unknown: Tensor):
        is_for_num, ptr, cat_ptr = False, 0, 0
        info_list = []
        while ptr < self._unknown_dim:
            if cat_ptr < len(self._cat_dims) and ptr == self._cat_dims[cat_ptr][0]:
                l, r = self._cat_dims[cat_ptr]
                cat_ptr += 1
                if r - l > 1 and not is_for_num:
                    info_list.append([SpanInfo(r-l, 'softmax')])
                elif is_for_num:
                    info_list.append([SpanInfo(1, 'tanh'), SpanInfo(r-l, 'softmax')])
                else:
                    info_list.append([SpanInfo(1, 'sigmoid')])
                is_for_num = False
                ptr = r
            else:
                ptr += 1
                is_for_num = True
        self._sampler = DataSampler(
            data=unknown,
            context=known,
            info=info_list
        )

    def _collate_fn(self, batch: List[Tuple[Tensor, ...]]) -> Tuple[Tensor, ...]:
        pass

    def _collate_fn_infer(self, batch: List[Tuple[Tensor, ...]]) -> Tuple[Tensor, ...]:
        pass

    def train(self, known: Tensor, unknown: Tensor, epochs: int = 10, batch_size: int = 100, shuffle: bool = True,
              save_freq: int = 100, resume: bool = True, lae_epochs: int = 10):
        self._construct_sampler(known, unknown)
        super().train(known, unknown, epochs, batch_size, shuffle, save_freq, resume, lae_epochs)

    def run_step(self, batch: Tuple[Tensor, ...]) -> Tuple[Dict[str, float], Optional[Tensor]]:
        pass

    @torch.no_grad()
    def inference(self, known: Tensor, batch_size: int) -> InferenceOutput:
        pass
