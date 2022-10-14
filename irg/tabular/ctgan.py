"""Partial CTGAN Training."""

from collections import OrderedDict
import math
from typing import Tuple, Dict, Optional, Any, List
import os

import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.cuda.amp import GradScaler
from packaging import version
from ctgan.synthesizers.ctgan import Generator
from tqdm import tqdm

from .base import TabularTrainer
from ..utils import InferenceOutput
from ..utils.dist import is_main_process
from ..utils.torch import Discriminator


class CTGANOutput(InferenceOutput):
    def __init__(self, fake: Tensor, discr_out: Optional[Tensor] = None):
        super().__init__(fake)
        self.fake = fake
        """Fake data generated."""
        self.discr_out = discr_out
        """Discriminator output."""


class CTGANTrainer(TabularTrainer):
    """Trainer for CTGAN."""
    def __init__(self, cond_cat_range: Tuple[int, int] = (3, 10), embedding_dim: int = 128,
                 generator_dim: Tuple[int, ...] = (256, 256), discriminator_dim: Tuple[int, ...] = (256, 256),
                 pac: int = 10, discriminator_step: int = 1, **kwargs):
        """
        **Args**:

        - `cond_cat_range` (`Tuple[int, int]`): For known dimension = 0, generation of condvec based on CTGAN,
          only on categories with number of categories in this range. For example, if there are 3 categorical values
          with number of categories 2, 5, 20 respectively, and given default value for this parameter (3, 10),
          the only possible choice on condvec construction is based on the second categorical column. In the case that
          no categorical column satisfy the range constraint, we do not use any condvec.
        - `embedding_dim` to `discriminator_step`: Arguments for
          [CTGAN](https://sdv.dev/SDV/api_reference/tabular/api/sdv.tabular.ctgan.CTGAN.html#sdv.tabular.ctgan.CTGAN).
        - `kwargs`: It has the following groups:
            - Inherited arguments from [`TabularTrainer`](./base#irg.tabular.base.TabularTrainer).
            - Generator arguments, all prefixed with "gen_", and others are the same as arguments for AutoEncoder for
              the parent class [`TabularTrainer`](./base#irg.tabular.base.TabularTrainer).
            - Discriminator arguments, all prefixed with "disc_".
              as generator.
        """
        super().__init__(**{
            n: v for n, v in kwargs.items() if
            n in {'distributed', 'autocast', 'log_dir', 'ckpt_dir', 'descr',
                  'cat_dims', 'known_dim', 'unknown_dim'}
        })

        if os.path.exists(self._aux_info_path):
            self._condvec_left, self._condvec_right, self._condvec_dim, self._condvec_accumulated = \
                torch.load(self._aux_info_path)
        else:
            self._condvec_left, self._condvec_right = 0, 0
            if self._known_dim == 0:
                rearranged_cat_dims_idx = np.random.choice(range(len(self._cat_dims)), len(self._cat_dims),
                                                           replace=False)
                left, right = cond_cat_range
                for i in rearranged_cat_dims_idx:
                    l, r = self._cat_dims[i]
                    if left <= r - l <= right:
                        self._condvec_left, self._condvec_right = l, r
                        break
            self._condvec_dim = self._condvec_right - self._condvec_left
            self._condvec_accumulated = [0 for _ in range(self._condvec_dim)]
            torch.save((self._condvec_left, self._condvec_right, self._condvec_dim, self._condvec_accumulated),
                       self._aux_info_path)

        self._encoded_dim = 0 if self._known_dim == 0 else self._lae.encoded_dim
        self._generator = Generator(
            embedding_dim=embedding_dim + self._encoded_dim + self._condvec_dim,
            generator_dim=generator_dim,
            data_dim=self._unknown_dim
        ).to(self._device)
        self._discriminator = Discriminator(
            input_dim=self._encoded_dim + self._condvec_dim + self._unknown_dim,
            discriminator_dim=discriminator_dim,
            pac=pac
        ).to(self._device)
        self._generator, self._optimizer_g, self._lr_schd_g, self._grad_scaler_g = self._make_model_optimizer(
            self._generator,
            **{n[4:]: v for n, v in kwargs.items() if n.startswith('gen_')}
        )
        self._discriminator, self._optimizer_d, self._lr_schd_d, self._grad_scaler_d = self._make_model_optimizer(
            self._discriminator,
            **{n[5:]: v for n, v in kwargs.items() if n.startswith('disc_')}
        )

        self._embedding_dim, self._pac, self._discriminator_step = embedding_dim, pac, discriminator_step

    @property
    def _aux_info_path(self) -> str:
        return os.path.join(self._ckpt_dir, self._descr, 'info.pt')

    def _load_content_from(self, loaded: Dict[str, Any]):
        super()._load_content_from(loaded)
        self._generator, self._optimizer_g, self._lr_schd_g, self._grad_scaler_g = self._load_state_dict(
            self._generator, self._optimizer_g, self._lr_schd_g, self._grad_scaler_g, loaded['generator']
        )
        self._discriminator, self._optimizer_d, self._lr_schd_d, self._grad_scaler_d = self._load_state_dict(
            self._discriminator, self._optimizer_d, self._lr_schd_d, self._grad_scaler_d, loaded['discriminator']
        )
        self._condvec_accumulated, self._condvec_left, self._condvec_right, self._condvec_dim = loaded['condvec']

    def _construct_content_to_save(self) -> Dict[str, Any]:
        return {
            'generator': self._full_state_dict(
                self._generator, self._optimizer_g, self._lr_schd_g, self._grad_scaler_g
            ),
            'discriminator': self._full_state_dict(
                self._discriminator, self._optimizer_d, self._lr_schd_d, self._grad_scaler_d
            ),
            'condvec': (self._condvec_accumulated, self._condvec_left, self._condvec_right, self._condvec_dim)
        } | super()._construct_content_to_save()

    def run_step(self, known: Tensor, unknown: Tensor) -> Tuple[Dict[str, float], Optional[Tensor]]:
        mean = torch.zeros(known.shape[0], self._embedding_dim, device=self._device)
        std = mean + 1
        enable_autocast = torch.cuda.is_available() and self._autocast
        condvec = unknown[:, self._condvec_left:self._condvec_right]
        if self._condvec_dim > 0:
            conditions = condvec.argmax(dim=1)
            for v in conditions:
                self._condvec_accumulated[v.item()] += 1
        known = self._make_context(known)

        for ds in range(self._discriminator_step):
            with torch.cuda.amp.autocast(enabled=enable_autocast):
                fake_cat = self._construct_fake(mean, std, known)
                real_cat = torch.cat([known, condvec, unknown], dim=1)
                y_fake, y_real = self._discriminator(fake_cat), self._discriminator(real_cat)
                pen = self._discriminator.calc_gradient_penalty(
                    real_cat, fake_cat, self._device, self._pac
                ) / (known.shape[1] + self._embedding_dim)
                loss_d = -(torch.mean(y_real) - torch.mean(y_fake))
                if self._grad_scaler_d is None:
                    pen.backward(retain_graph=True)
                else:
                    self._grad_scaler_d.scale(pen).backward(retain_graph=True)
            self._take_step(loss_d, self._optimizer_d, self._grad_scaler_d, self._lr_schd_d)
            # TODO: for param in model.parameters(): param.grad = None

        with torch.cuda.amp.autocast(enabled=enable_autocast):
            fake_cat = self._construct_fake(mean, std, known)
            real_cat = torch.cat([known, condvec, unknown], dim=1)
            y_fake = self._discriminator(fake_cat)
            if known.shape[1] == 0:
                distance = torch.tensor(0)
            else:
                distance = F.mse_loss(fake_cat, real_cat, reduction='mean')
            meta_loss = self._meta_loss(known, unknown, fake_cat[:, -self.unknown_dim:])
            loss_g = -torch.mean(y_fake) + distance + meta_loss
        self._take_step(loss_g, self._optimizer_g, self._grad_scaler_g, self._lr_schd_g)
        return {
                   'G loss': loss_g.detach().cpu().item(),
                   'meta': meta_loss.detach().cpu().item(),
                   'D loss': loss_d.detach().cpu().item(),
                   'distance': distance.detach().cpu().item(),
                   'penalty': pen.detach().cpu().item(),
                   'G lr': self._optimizer_g.param_groups[0]['lr'],
                   'D lr': self._optimizer_d.param_groups[0]['lr']
               }, fake_cat

    def _construct_fake(self, mean: Tensor, std: Tensor, known_tensor: Tensor) -> Tensor:
        fakez = torch.normal(mean=mean, std=std)[:known_tensor.shape[0]].to(self._device)
        if self._condvec_dim > 0:
            sum_cnt = sum(self._condvec_accumulated)
            probabilities = [x / sum_cnt for x in self._condvec_accumulated]
            conditions = np.random.choice(range(self._condvec_dim), known_tensor.shape[0], p=probabilities)
            condvec = F.one_hot(torch.from_numpy(conditions).long(), self._condvec_dim).to(self._device)
        else:
            condvec = known_tensor[:, 0:0]
        fakez = torch.cat([fakez, condvec, known_tensor], dim=1)
        fake = self._generator(fakez)
        fakeact = self._apply_activate(fake)
        fake_cat = torch.cat([known_tensor, condvec, fakeact], dim=1)
        return fake_cat

    @classmethod
    def _reconstruct(cls, distributed: bool, autocast: bool, log_dir: str, ckpt_dir: str, descr: str,
                     known_dim: int, unknown_dim: int, cat_dims: List[Tuple[int, int]], lae_trained: bool,
                     lae: nn.Module, optimizer_lae: Optimizer, lr_schd_lae: LRScheduler,
                     grad_scaler_lae: Optional[GradScaler],
                     condvec_left: int, condvec_right: int, condvec_dim: int, condvec_accumulated: List[int],
                     generator: nn.Module, optimizer_g: Optimizer, lr_schd_g: LRScheduler,
                     grad_scaler_g: Optional[GradScaler], discriminator: nn.Module, optimizer_d: Optimizer,
                     lr_schd_d: LRScheduler, grad_scaler_d: Optional[GradScaler], lae: nn.Module,
                     optimizer_l: Optimizer, lr_schd_l: LRScheduler, grad_scaler_l: Optional[GradScaler],
                     embedding_dim: int, pac: int, discriminator_step: int, encoded_dim: int) -> "CTGANTrainer":
        base = TabularTrainer._reconstruct(
            distributed, autocast, log_dir, ckpt_dir, descr,
            known_dim, unknown_dim, cat_dims, lae_trained,
            lae, optimizer_lae, lr_schd_lae, grad_scaler_lae
        )
        base.__class__ = CTGANTrainer
        base._condvec_left, base._condvec_right, base._condvec_dim, base._condvec_accumulated = (
            condvec_left, condvec_right, condvec_dim, condvec_accumulated
        )
        base._generator, base._optimizer_g, base._lr_schd_g, base._grad_scaler_g = (
            generator, optimizer_g, lr_schd_g, grad_scaler_g)
        base._discriminator, base._optimizer_d, base._lr_schd_d, base._grad_scaler_d = (
            discriminator, optimizer_d, lr_schd_d, grad_scaler_d)
        base._lae, base._optimizer_l, base._lr_schd_l, base._grad_scaler_l = (
            lae, optimizer_l, lr_schd_l, grad_scaler_l)
        base._embedding_dim, base._pac, base._discriminator_step = embedding_dim, pac, discriminator_step
        base._encoded_dim = encoded_dim
        return base

    def __reduce__(self):
        _, var = super().__reduce__()
        return self._reconstruct, var + (
            self._condvec_left, self._condvec_right, self._condvec_dim, self._condvec_accumulated,
            self._generator, self._optimizer_g, self._lr_schd_g, self._grad_scaler_g,
            self._discriminator, self._optimizer_d, self._lr_schd_d, self._grad_scaler_d,
            self._embedding_dim, self._pac, self._discriminator_step, self._encoded_dim
        )

    def _apply_activate(self, data: Tensor) -> Tensor:
        act_data, mask, ptr, cat_ptr = [], [], 0, 0
        while ptr < self._unknown_dim:
            if cat_ptr < len(self._cat_dims) and ptr == self._cat_dims[cat_ptr][0]:
                l, r = self._cat_dims[cat_ptr]
                cat_ptr += 1
                if r - l == 1:
                    act_data.append(torch.sigmoid(data[:, l:r]))
                else:
                    act_data.append(self._gumbel_softmax(data[:, l:r], tau=0.2))
                ptr = r
                mask.append(torch.ones(data.shape[0], r-l).to(self._device))
            else:
                act_data.append(torch.tanh(data[:, ptr:ptr+1]))
                ptr += 1
                mask.append(torch.zeros(data.shape[0], 1).to(self._device))
        activated = torch.cat(act_data, dim=1)
        return activated

    @staticmethod
    def _gumbel_softmax(logits: torch.Tensor, tau: float = 1, hard: bool = False,
                        eps: float = 1e-10):
        if version.parse(torch.__version__) < version.parse('1.2.0'):
            for i in range(10):
                transformed = F.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=-1)
                if not torch.isnan(transformed).any():
                    return transformed
            raise ValueError('gumbel_softmax returning NaN.')
        return F.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=-1)

    @torch.no_grad()
    def inference(self, known: Tensor, batch_size: int) -> CTGANOutput:
        mean = torch.zeros(known.shape[0], self._embedding_dim, device=self._device)
        std = mean + 1

        dataloader = self._make_dataloader(known, torch.zeros(known.shape[0], self._unknown_dim), batch_size, False)
        if is_main_process():
            dataloader = tqdm(dataloader)
            dataloader.set_description(f'Inference on {self._descr}')

        fakes, y_fakes = [], []
        self._generator.eval()
        self._discriminator.eval()
        for step, (known_batch, _) in enumerate(dataloader):
            known_batch = known_batch.to(self._device)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available() and self._autocast):
                known = self._make_context(known_batch)
                fake_cat = self._construct_fake(mean, std, known)
                y_fake = self._discriminator(fake_cat)
                y_fake = y_fake.repeat(self._pac, 1).permute(1, 0).flatten()[:fake_cat.shape[0]]
                fakes.append(fake_cat)
                y_fakes.append(y_fake)
        self._generator.train()
        self._discriminator.train()

        return CTGANOutput(torch.cat(fakes), torch.cat(y_fakes))
