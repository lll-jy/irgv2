from collections import OrderedDict
from typing import Tuple, Dict, Optional
import os

import torch
from torch import Tensor
import torch.nn.functional as F
from tqdm import tqdm

from .base import TabularTrainer
from ..utils import InferenceOutput
from ..utils.torch import MLP
from ..utils.dist import is_main_process


class MLPTrainer(TabularTrainer):
    """Trainer for MLP"""
    def __init__(self, err_retain: int = 100, **kwargs):
        """
        **Args**:

        - `err_retain` (`int`): Number of instances to retain for approximating the errors.
        - `kwargs`: It has the following groups:
            - Inherited arguments from [`TabularTrainer`](./base#irg.tabular.base.TabularTrainer).
            - Model arguments to [`MLP`](../utils/torch#irg.utils.torch.MLP).
        """
        super().__init__(**{
            n: v for n, v in kwargs.items() if
            n in {'distributed', 'autocast', 'log_dir', 'ckpt_dir', 'descr',
                  'cat_dims', 'known_dim', 'unknown_dim'}
        })
        self._mlp = MLP(**{
            n: v for n, v in kwargs.items() if
            n in {'in_dim', 'out_dim', 'hidden_dim', 'dropout', 'act'}
        })
        (self._mlp,), self._optimizer, self._lr_schd, self._grad_scaler = self._make_model_optimizer(
            [self._mlp],
            **{n: v for n, v in kwargs.items() if
               n not in {'distributed', 'autocast', 'log_dir', 'ckpt_dir', 'descr',
                         'cat_dims', 'known_dim', 'unknown_dim', 'in_dim', 'out_dim', 'hidden_dim', 'dropout', 'act'}}
        )
        self._err_retain = err_retain
        self._err = []

    def _reload_checkpoint(self, idx: int, by: str):
        path = os.path.join(self._ckpt_dir, self._descr, f'{by}_{idx:07d}.pt')
        if not os.path.exists(path):
            return
        loaded = torch.load(path)
        mlp_dict = loaded['mlp']
        if hasattr(self._mlp, 'module'):
            mlp_dict = OrderedDict({f'module.{n}': v for n, v in mlp_dict.items()})
        self._mlp.load_state_dict(mlp_dict, strict=True)
        self._optimizer.load_state_dict(loaded['optimizer'])
        self._lr_schd.load_state_dict(loaded['lr_scheduler'])
        self._grad_scaler.load_state_dict(loaded['grad_scaler'])
        torch.manual_seed(loaded['seed'])

    def _save_checkpoint(self, idx: int, by: str):
        torch.save(
            {
                'mlp': (self._mlp.module if hasattr(self._mlp, 'module') else self._mlp).state_dict(),
                'optimizer': self._optimizer.state_dict(),
                'lr_scheduler': self._lr_schd.state_dict(),
                'grad_scaler': self._grad_scaler.state_dict(),
                'seed': torch.initial_seed()
            },
            os.path.join(self._ckpt_dir, self._descr, f'{by}_{idx:07d}.pt')
        )

    def run_step(self, known: Tensor, unknown: Tensor) -> Tuple[Dict[str, float], Optional[Tensor]]:
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available() and self._autocast):
            pred = self._mlp(known)
            loss = self._calc_loss(pred, unknown)
            self._err += list(pred.split(1))
            self._err = self._err[-self._err_retain:]
        self._take_step(loss, self._optimizer, self._grad_scaler, self._lr_schd)
        return {
            'Loss': loss.detach().cpu().item()
        }, pred

    def _calc_loss(self, pred: Tensor, truth: Tensor) -> Tensor:
        loss, ptr, cat_ptr = torch.tensor(0), 0, 0
        while ptr < self._unknown_dim:
            if cat_ptr < len(self._cat_dims) and ptr == self._cat_dims[cat_ptr][0]:
                l, r = self._cat_dims[cat_ptr]
                cat_ptr += 1
                if r - l == 1:
                    loss += F.binary_cross_entropy(
                        pred[:, l:r].clamp(0, 1), truth[:, l:r].clamp(0, 1), reduction='sum')
                else:
                    loss += F.cross_entropy(
                        pred[:, l:r], torch.argmax(torch[:, l:r], dim=-1), reduction='sum')
                ptr = r
            else:
                loss += F.mse_loss(pred, truth, reduction='sum')
                ptr += 1
        return loss / pred.shape[0]

    @torch.no_grad()
    def inference(self, known: Tensor, batch_size: int) -> InferenceOutput:
        dataloader = self._make_dataloader(known, torch.zeros(known.shape[0], self._unknown_dim), batch_size, False)
        if is_main_process():
            dataloader = tqdm(dataloader)
            dataloader.set_description(f'Inference on {self._descr}')

        errors = torch.stack(self._err)
        mu, std = errors.mean(dim=-1), errors.std(dim=-1)

        predictions = []
        self._mlp.eval()
        for step, (known_batch, _) in enumerate(dataloader):
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available() and self._autocast):
                pred = self._mlp(known)
                err = torch.normal(mu, std)
                predictions.append(pred + err)
        self._mlp.train()

        return InferenceOutput(torch.stack(predictions))
