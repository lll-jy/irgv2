import math
import os
from typing import List, Optional, Dict, Tuple, Literal

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
import torch
from torch import Tensor
from torch.nn import Sequential, Linear, ReLU, Dropout, MSELoss, Tanh, Identity
from torch.optim import Adam, Optimizer
from tqdm import tqdm

from .base import DegreeTrainer
from ..schema import Table, Database, SyntheticTable, SyntheticDatabase
from ..utils.misc import make_dataloader, convert_data_as
from ..utils.dist import get_device, to_device
from ..utils.io import load_state_dict, save_state_dict


class DegreeSteppedTrainer(DegreeTrainer):
    """Degree prediction by each foreign key one by one."""
    def __init__(self, lr: float = 0.001, n_epoch: int = 10, batch_size: int = 64, top_k: float = 0.5,
                 hidden_dim: int = 100, n_layer: int = 3, dropout: float = 0.1, **kwargs):
        """
        **Args**:

        - `lr` (`float`): Learning rate. Default is 0.001.
        - `n_epoch` (`int`): Number of epochs to train. Default is 10.
        - `batch_size` (`int`): Batch size during training. Default is 64.
        - `top_k` (`float`): Additional proportion to choose top k from. Default is 0.5.
          This means to select k nearly most probable items, we select (1 + top_k) * k items,
          and sample k from them based on their scores.
        - `hidden_dim` (`int`): Number of hidden dimensions for models. Default is 100.
        - `n_layer` (`int`): Number of layers for models. Default is 3.
        - `dropout` (`float`): Dropout of the models. Default is 0.1.
        """
        super().__init__(**kwargs)
        self._lr = lr
        self._n_epoch = n_epoch
        self._batch_size = batch_size
        self._top_k = top_k
        self._hidden_dim = hidden_dim
        self._n_layer = n_layer
        self._dropout = dropout
        self._model_size = [(0, 0)] * len(self._foreign_keys)  # (input size, table size)
        self._norm = [StandardScaler() for _ in self._foreign_keys]
        os.makedirs(os.path.join(self._cache_dir, 'models'), exist_ok=True)
        self._device = get_device()
        self._real_sum = [0] * len(self._foreign_keys)

    def _model_path_for(self, idx: int) -> str:
        return os.path.join(self._cache_dir, 'models', f'fk{idx}_model.pt')

    def _match_path_for(self, idx: int) -> str:
        return os.path.join(self._cache_dir, 'models', f'fk{idx}_match.pt')

    def _fit(self, data: Table, context: Database):
        # 1. fit degrees of combination thus far
        # 2. fit matching model from combination thus far to inverse embedding
        deg_data = data.data('degree', normalize=False, with_id='this')
        deg_norm = data.data('degree', normalize=True, with_id='none')
        train_cols = []
        fk_cols = []
        for i, fk in enumerate(self._foreign_keys):
            train_cols.extend([(a, b, c) for a, b, c in deg_norm.columns if a.startswith(f'fk{i}:')])
            train_data = deg_norm[train_cols]
            fk_cols.extend(fk.left)
            grouped = deg_data[fk_cols + [('', 'degree')]].groupby(fk_cols, as_index=False).sum()
            self._real_sum[i] = grouped[('', 'degree')].sum()
            degrees = deg_data[fk_cols + [('', 'degree')]].groupby(fk_cols).transform('sum')[('', 'degree')]
            assert len(degrees) == len(train_data), f'Size of prediction models must match, ' \
                                                    f'got {len(degrees)} != {len(train_data)} in' \
                                                    f' {self._name}:{fk.parent}-{fk.child}'
            norm_from = grouped[('', 'degree')]
            if i > 0:
                norm_from = norm_from[norm_from > 0]
            self._norm[i].fit(norm_from.to_numpy().reshape(-1, 1))
            self._fit_pred_model(train_data, degrees, i)

            if i == len(self._foreign_keys) - 1:
                return
            parent_normalized = context.augmented_till(self._foreign_keys[i+1].parent, data.name, with_id='none')
            columns = [(f'fk{i+1}:{self._foreign_keys[i+1].parent}', a, b) for a, b in parent_normalized.columns]
            self._fit_matching(train_data, deg_norm[columns], i)

    def _make_mlp(self, in_dim: int, out_dim: int, last_act: Literal['id', 'tanh'], path: str) -> Sequential:
        in_dim = [in_dim] + [self._hidden_dim] * (self._n_layer - 1)
        out_dim = [self._hidden_dim] * (self._n_layer - 1) + [out_dim]
        act = [ReLU() for _ in range(self._n_layer - 1)] + [Tanh() if last_act == 'tanh' else Identity()]
        model = Sequential(*[
            Sequential(
                Linear(i, o),  # layer
                Dropout(self._dropout),  # dropout
                a  # activation
            ) for i, o, a in zip(in_dim, out_dim, act)
        ])
        if os.path.exists(path):
            load_state_dict(model, path)
        return model

    def _make_pred_model(self, idx: int) -> Sequential:
        return self._make_mlp(self._model_size[idx][0], 1, 'id', self._model_path_for(idx))

    def _fit_pred_model(self, x: pd.DataFrame, y: pd.Series, idx: int):
        self._model_size[idx] = x.shape[-1], self._model_size[idx][1]
        model = self._make_pred_model(idx)
        if os.path.exists(self._model_path_for(idx)):
            return
        optim = Adam(model.parameters(), lr=self._lr)
        criterion = MSELoss()
        y = pd.DataFrame({'degree': self._norm[idx].transform(y.to_numpy().reshape(-1, 1)).reshape(-1)})
        x = convert_data_as(x, 'torch').float()
        y = convert_data_as(y, 'torch').float()
        model, x, y, model, criterion, optim = to_device((model, x, y, model, criterion, optim), self._device)
        dataloader = make_dataloader(x, y, batch_size=self._batch_size)
        for e in range(self._n_epoch):
            losses = []
            iterator = tqdm(dataloader, desc=f'DegFk{idx} of {self._name} Epoch[{e}]: Loss: 0.000, Avg: 0.000')
            for ctx, deg in iterator:
                out = model(ctx)
                loss = criterion(out, deg)
                losses.append(loss)
                self._wrap_step(optim, loss, iterator, losses, f'DegFk{idx} of {self._name} Epoch[{e}]')
        save_state_dict(model, self._model_path_for(idx))

    @torch.no_grad()
    def _infer_pred_model(self, x: pd.DataFrame, idx: int) -> pd.Series:
        model = self._make_pred_model(idx)
        x = convert_data_as(x, 'torch')
        model, x, model = to_device((model, x, model), self._device)
        dataloader = make_dataloader(x.float(), batch_size=self._batch_size, shuffle=False)
        results = []
        iterator = tqdm(dataloader, desc=f'DegFk{idx} of {self._name} Infer')
        for ctx, in iterator:
            out = model(ctx)
            results.append(out)
        results = torch.cat(results).view(-1)

        scaler = StandardScaler()
        fkcomb_degrees = scaler.fit_transform(results.view(-1, 1).numpy())
        fkcomb_degrees = pd.Series(self._norm[idx].inverse_transform(fkcomb_degrees).reshape(-1))
        return fkcomb_degrees

    def _make_match_model(self, idx: int) -> Sequential:
        return self._make_mlp(self._model_size[idx][0], self._model_size[idx][1], 'tanh', self._match_path_for(idx))

    def _fit_matching(self, x: pd.DataFrame, target: pd.DataFrame, idx: int):
        self._model_size[idx] = self._model_size[idx][0], target.shape[-1]
        model = self._make_match_model(idx)
        if os.path.exists(self._match_path_for(idx)):
            return

        optim = Adam(model.parameters(), lr=self._lr)
        criterion = MSELoss()
        x = convert_data_as(x, 'torch').float()
        y = convert_data_as(target, 'torch').float()
        model, criterion, x, y, optim = to_device(
            (model, criterion, x, y, optim), self._device)
        dataloader = make_dataloader(x, y, batch_size=self._batch_size, shuffle=True)

        for e in range(self._n_epoch):
            iterator = tqdm(dataloader,
                            desc=f'Training fk{idx} in {self._name} Epoch [{e}]: Loss: 0.000, Avg: 0.000')
            losses = []
            for ctx, next_data in iterator:
                out = model(ctx)
                loss = criterion(out, next_data)
                losses.append(loss)

                self._wrap_step(optim, loss, iterator, losses, f'Training fk{idx} in {self._name} Epoch [{e}]')
        save_state_dict(model, self._match_path_for(idx))

    @staticmethod
    def _normalize_inverse_prob(v: Tensor) -> Tensor:
        if v.shape[-1] == 1:
            return v / v
        v = (-v).softmax(dim=-1)
        v = (v + (1 - v.sum()) / v.shape[-1]).clamp(min=0., max=1.)
        if v.sum() == 1:
            return v
        diff = v.sum() - 1
        if diff < 0:
            can = v < 1 + diff
        else:
            can = v > diff
        can = can.nonzero().flatten()
        idx = np.random.choice(can)
        v[idx] -= diff
        return v

    @torch.no_grad()
    def _infer_matching(self, x: pd.DataFrame, next_data: pd.DataFrame, degrees: pd.Series, idx: int) \
            -> (pd.DataFrame, List[Tuple[int, int]]):
        model = self._make_match_model(idx)

        x_columns = [*x.columns]
        x = convert_data_as(x, 'torch').float()
        next_df = next_data
        next_data = convert_data_as(next_data, 'torch').float()
        degrees = convert_data_as(pd.DataFrame({'degree': degrees}), 'torch').float()
        model, x, next_data, degrees = to_device((model, x, next_data, degrees), self._device)
        dataloader = make_dataloader(x, degrees, batch_size=self._batch_size, shuffle=False)

        iterator = tqdm(enumerate(dataloader), total=len(dataloader),
                        desc=f'Infer matching fk{idx} in {self._name}: Size 0')
        results = []
        matches = []
        for step, (ctx, deg) in iterator:
            out = model(ctx)
            pairwise_euclidean = euclidean_distances(out, next_data)
            values, indices = torch.topk(
                input=torch.tensor(pairwise_euclidean),
                k=min(max(math.ceil(deg.max().item() * (1 + self._top_k)), 0), next_data.shape[0]),
                dim=-1)
            for k, c, v, i, d in zip(range(len(deg)), ctx, values, indices, deg):
                c = pd.Series(c, index=x_columns)
                total = math.ceil(d.item() * (1 + self._top_k))
                if total <= 0:
                    continue
                v = self._normalize_inverse_prob(v[:total]).numpy()
                chosen = np.random.choice(
                    a=i[:total].numpy(),
                    size=max(1, min(max(int(d.item()), 1), next_data.shape[0], (v != 0).sum())),
                    replace=False,
                    p=v
                )
                results.append(pd.concat([
                        c.to_frame().T.iloc[[0] * len(chosen)].reset_index(drop=True),
                        next_df.iloc[chosen].reset_index(drop=True)
                ], axis=1))
                for j in chosen:
                    matches.append((step * self._batch_size + k, j))
                iterator.set_description(f'Infer matching fk{idx} in {self._name}: Size {len(matches)}')
        results = pd.concat(results)
        return results, matches

    @staticmethod
    def _wrap_step(optim: Optimizer, loss: Tensor, iterator: tqdm, losses: List[Tensor], prefix: str):
        optim.zero_grad()
        loss.backward()
        optim.step()

        iterator.set_description(
            f'{prefix}: '
            f'Loss: {loss.item():.3f}, Avg: {sum(losses).item() / len(losses):.3f}'
        )

    def predict(self, data: SyntheticTable, context: SyntheticDatabase, scaling: Optional[Dict[str, float]],
                tolerance: float = 0.05) -> (Tensor, pd.DataFrame):
        # 1. predict degrees of combination thus far
        # 2. if not the last one, predict next round context
        # 3. construct next context
        current_context = context.augmented_till(self._foreign_keys[0].parent, self._name, with_id='none')
        known_so_far = context.augmented_till(
            self._foreign_keys[0].parent, self._name, with_id='this', normalized=False)
        known_so_far = pd.concat({f'fk0:{self._foreign_keys[0].parent}': known_so_far}, axis=1)
        for i, fk in enumerate(self._foreign_keys):
            assert len(current_context) == len(known_so_far)
            fkcomb_degrees = self._infer_pred_model(current_context, i)
            # factor *= scaling[fk.parent]
            factor = scaling[fk.parent]
            fkcomb_degrees = fkcomb_degrees / factor
            real = self._real_sum[i]
            fkcomb_degrees, _ = self._round_sumrange(fkcomb_degrees, real * (1 - tolerance), real * (1 + tolerance),
                                                     till_in_range=True)
            print('so I want here', self._name, self._real_sum[i], factor, real, 'got', fkcomb_degrees.sum())

            known_so_far[fk.left] = known_so_far[[(f'fk{i}:{t}', c) for t, c in fk.right]]

            if i == len(self._foreign_keys) - 1:
                pred_deg = fkcomb_degrees * scaling[self._name]
                real = real * scaling[self._name]
                pred_deg, _ = self._round_sumrange(pred_deg, real * (1 - tolerance), real * (1 + tolerance),
                                                   till_in_range=True)
                print('finally scale', self._name, scaling[self._name], real, 'got', pred_deg.sum(), flush=True)
                break
            
            next_table = context.augmented_till(self._foreign_keys[i+1].parent, self._name, with_id='none')
            current_context, matches = self._infer_matching(current_context, next_table, fkcomb_degrees, i)
            next_table_raw = context.augmented_till(self._foreign_keys[i+1].parent, self._name,
                                                    with_id='this', normalized=False)
            next_table_raw = next_table_raw.rename(columns={
                c: (f'fk{i+1}:{self._foreign_keys[i+1].parent}', c)
                for c in next_table_raw.columns
            })
            known_id, next_id = [], []
            for x, y in matches:
                known_id.append(x)
                next_id.append(y)
            known_so_far = pd.concat([
                known_so_far.iloc[known_id].reset_index(drop=True),
                next_table_raw.iloc[next_id].reset_index(drop=True)],
                axis=1)

        data.save_degree_known(known_so_far)
        data.assign_degrees(pred_deg)
        known_tab, _, _ = data.ptg_data()
        augmented = data.data('augmented')
        return known_tab, augmented
