from collections import OrderedDict
from typing import Any, Dict, Tuple, Optional

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from .ctgan import CTGANTrainer
from ..utils.torch import CNNDiscriminator


class MTGANTrainer(CTGANTrainer):
    """Trainer for MTGAN."""
    def __init__(self, meta_samples: int = 10, meta_size: int = 50, **kwargs):
        """
        **Args**:

        - `meta_samples` (`int`): Number of samples to feed in the batch-wise discriminator in each step, for real and
          fake respectively. Default is 10.
        - `meta_size` (`int`): Number of rows to construct each group of input to batch-wise discriminator.
          Default is 50.
        - `kwargs`: It has the following groups:
            - Inherited arguments from [`CTGANTrainer`](./ctgan#irg.tabular.ctgan.CTGANTrainer).
            - Arguments for [`CNNDiscriminator`](../utils/torch#irg.utils.torch.CNNDiscriminator) as batch-wise
              discriminator. All prefixed with "cnn_model_".
            - Arguments for optimizers and schedulers for the batch-wise discriminator, all prefixed with "cnn_".
        """
        super_args = {n: v for n, v in kwargs.items() if not n.startswith('cnn_')}
        cnn_args = {n[10:]: v for n, v in kwargs.items() if n.startswith('cnn_model_')}
        cnn_training_args = {n[4:]: v for n, v in kwargs.items()
                             if n.startswith('cnn_') and not n.startswith('cnn_model_')}
        super().__init__(**super_args)
        self._cnn = CNNDiscriminator(
            row_width=self._known_dim + self._unknown_dim,
            num_samples=meta_size,
            **cnn_args
        ).to(self._device)
        self._cnn, self._optimizer_cnn, self._lr_schd_cnn, self._grad_scaler_cnn = self._make_model_optimizer(
            self._cnn, **cnn_training_args
        )

        self._meta_samples, self._meta_size = meta_samples, meta_size

    def _load_content_from(self, loaded: Dict[str, Any]):
        super()._load_content_from(loaded)
        cnn_dict = loaded['cnn']['model']
        if hasattr(self._cnn, 'module'):
            cnn_dict = OrderedDict({f'module.{n}': v for n, v in cnn_dict.items()})
        self._cnn.load_state_dict(cnn_dict, strict=True)
        self._optimizer_cnn.load_state_dict(loaded['cnn']['optimizer'])
        self._lr_schd_cnn.load_state_dict(loaded['cnn']['lr_scheduler'])
        if 'grad_scaler' in loaded['cnn']:
            self._grad_scaler_cnn.load_state_dict(loaded['cnn']['grad_scaler'])
        else:
            self._grad_scaler_cnn = None

    def _construct_content_to_save(self) -> Dict[str, Any]:
        content = super()._construct_content_to_save()
        content['cnn'] = {
            'model': (self._cnn.module if hasattr(self._cnn, 'module')
                      else self._cnn).state_dict(),
            'optimizer': self._optimizer_cnn.state_dict(),
            'lr_scheduler': self._lr_schd_cnn.state_dict()
        } | ({'grad_scaler': self._grad_scaler_cnn.state_dict()} if self._grad_scaler_cnn is not None else {})
        return content

    def run_step(self, known: Tensor, unknown: Tensor) -> Tuple[Dict[str, float], Optional[Tensor]]:
        mean = torch.zeros(known.shape[0], self._embedding_dim, device=self._device)
        std = mean + 1
        enable_autocast = torch.cuda.is_available() and self._autocast
        for ds in range(self._discriminator_step):
            with torch.cuda.amp.autocast(enabled=enable_autocast):
                fake_cat = self._construct_fake(mean, std, known)
                real_cat = torch.cat([known, unknown], dim=1)
                y_fake, y_real = self._discriminator(fake_cat), self._discriminator(real_cat)
                pen = self._discriminator.calc_gradient_penalty(
                    real_cat, fake_cat, self._device, self._pac
                ) / (known.shape[1] + self._embedding_dim)
                loss_d = -(torch.mean(y_real) - torch.mean(y_fake))
                if self._grad_scaler_d is None:
                    pen.backward(retain_graph=True)
                else:
                    self._grad_scaler_d.scale(pen).backward(retain_graph=True)

                batch_y_fake = self._cnn(self._construct_cnn_input(fake_cat))
                batch_y_real = self._cnn(self._construct_cnn_input(real_cat))
                loss_bd = -(torch.mean(batch_y_real) - torch.mean(batch_y_fake))
            self._take_step(loss_d, self._optimizer_d, self._grad_scaler_d, self._lr_schd_d, retain_graph=True)
            self._take_step(loss_bd, self._optimizer_cnn, self._grad_scaler_cnn, self._lr_schd_cnn)

        with torch.cuda.amp.autocast(enabled=enable_autocast):
            fake_cat = self._construct_fake(mean, std, known)
            real_cat = torch.cat([known, unknown], dim=1)
            y_fake = self._discriminator(fake_cat)
            if known.shape[1] == 0:
                distance = torch.tensor(0)
            else:
                distance = F.mse_loss(fake_cat, real_cat, reduction='mean')
            batch_y_fake = self._cnn(self._construct_cnn_input(fake_cat))
            loss_g = -torch.mean(y_fake) + distance - torch.mean(batch_y_fake)

        self._take_step(loss_g, self._optimizer_g, self._grad_scaler_g, self._lr_schd_g)
        return {
            'GL': loss_g.detach().cpu().item(),
            'dis': distance.detach().cpu().item(),
            'DL': loss_d.detach().cpu().item(),
            'pen': pen.detach().cpu().item(),
            'BDL': loss_bd.detach().cpu().item(),
            'G_lr': self._optimizer_g.param_groups[0]['lr'],
            'D_lr': self._optimizer_d.param_groups[0]['lr'],
            'BD_lr': self._optimizer_cnn.param_groups[0]['lr']
        }, fake_cat

    def _construct_cnn_input(self, batch: Tensor):
        samples = []
        for _ in range(self._meta_samples):
            indices = np.random.choice(range(len(batch)), self._meta_size)
            samples.append(batch[indices].unsqueeze(0))
        return torch.stack(samples)
