"""Partial CTGAN Training."""

from typing import Tuple, Dict, Optional, Any, List
import os

import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader
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
    def __init__(self, embedding_dim: int = 128,
                 generator_dim: Tuple[int, ...] = (256, 256), discriminator_dim: Tuple[int, ...] = (256, 256),
                 pac: int = 10, discriminator_step: int = 1, **kwargs):
        """
        **Args**:

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

        # if os.path.exists(self._aux_info_path):
        #     self._condvec_left, self._condvec_right, self._condvec_dim, self._condvec_accumulated = \
        #         torch.load(self._aux_info_path)
        # else:
        #     self._condvec_left, self._condvec_right = 0, 0
        #     if self._known_dim == 0:
        #         rearranged_cat_dims_idx = np.random.choice(range(len(self._cat_dims)), len(self._cat_dims),
        #                                                    replace=False)
        #         left, right = cond_cat_range
        #         for i in rearranged_cat_dims_idx:
        #             l, r = self._cat_dims[i]
        #             if left <= r - l <= right:
        #                 self._condvec_left, self._condvec_right = l, r
        #                 break
        #     self._condvec_dim = self._condvec_right - self._condvec_left
        #     self._condvec_accumulated = [0 for _ in range(self._condvec_dim)]
        #     torch.save((self._condvec_left, self._condvec_right, self._condvec_dim, self._condvec_accumulated),
        #                self._aux_info_path)
        if os.path.exists(self._aux_info_path):
            self._cond_dist, self._cond_span = torch.load(self._aux_info_path)
        else:
            self._cond_dist, self._cond_span = [], self._learn_cond_span()

        self._encoded_dim = 0 if self._known_dim == 0 else self._lae.encoded_dim
        context_dim = self._encoded_dim if self._encoded_dim > 0 else self._known_dim if len(self._cond_span) > 0 else 0
        self._generator = Generator(
            embedding_dim=embedding_dim + context_dim,
            generator_dim=generator_dim,
            data_dim=self._unknown_dim
        ).to(self._device)
        self._discriminator = Discriminator(
            input_dim=context_dim + self._unknown_dim,
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
        self._cond_dist, self._cond_span = loaded['condvec']
        # self._condvec_accumulated, self._condvec_left, self._condvec_right, self._condvec_dim = loaded['condvec']

    def _construct_content_to_save(self) -> Dict[str, Any]:
        return {
            'generator': self._full_state_dict(
                self._generator, self._optimizer_g, self._lr_schd_g, self._grad_scaler_g
            ),
            'discriminator': self._full_state_dict(
                self._discriminator, self._optimizer_d, self._lr_schd_d, self._grad_scaler_d
            ),
            'condvec': (self._cond_dist, self._cond_span)
            # 'condvec': (self._condvec_accumulated, self._condvec_left, self._condvec_right, self._condvec_dim)
        } | super()._construct_content_to_save()

    def _learn_cond_span(self):
        if self._known_dim == 0:
            is_for_num, ptr, cat_ptr = False, 0, 0
            while ptr < self._unknown_dim:
                if cat_ptr < len(self._cat_dims) and ptr == self._cat_dims[cat_ptr][0]:
                    l, r = self._cat_dims[cat_ptr]
                    cat_ptr += 1
                    if r - l > 1 and not is_for_num:
                        self._cond_span.append((l, r))
                    ptr = r
                else:
                    ptr += 1

    def _make_dataloader(self, known: Tensor, unknown: Tensor, batch_size: int, shuffle: bool = True) -> DataLoader:
        if self._known_dim == 0:
            for l, r in self._cond_span:
                cat_data = unknown[:, l:r]
                self._cond_dist.append(cat_data.sum(dim=0) / cat_data.sum())
        return super()._make_dataloader(known, unknown, batch_size, shuffle)

    def _sample_condvec_from(self, x: Tensor):
        if len(self._cond_span) == 0:
            return torch.zeros(x.shape[0], 0)
        cond_chosen = np.random.choice(range(len(self._cond_span)), x.shape[0], replace=True)
        masks = []
        for cond in cond_chosen:
            zeros = torch.zeros(x.shape[1])
            l, r = self._cond_span[cond]
            zeros[l:r] = 1
            masks.append(zeros)
        masks = torch.stack(masks)
        return x * masks

    def _sample_condvec(self, n: int):
        if len(self._cond_span) == 0:
            return torch.zeros(n, 0)
        cond_chosen = np.random.choice(range(len(self._cond_span)), n, replace=True)
        condvecs = []
        for cond in cond_chosen:
            zeros = torch.zeros(self._unknown_dim)
            l, r = self._cond_span[cond]
            opt = np.random.choice(range(r-l), p=self._cond_dist[cond])
            zeros[l+opt] = 1
            condvecs.append(zeros)
        return torch.stack(condvecs)

    def _collate_fn(self, batch: List[Tuple[Tensor, ...]]):
        known, unknown = super()._collate_fn(batch)
        condvec = self._sample_condvec_from(unknown)
        random_condvec = self._sample_condvec(unknown.shape[0])
        mean = torch.zeros(known.shape[0], self._embedding_dim, device=self._device)
        std = mean + 1
        noise1 = torch.normal(mean=mean, std=std)
        noise2 = torch.normal(mean=mean, std=std)
        return known, unknown, condvec, random_condvec, noise1, noise2

    def run_step(self, batch: Tuple[Tensor, ...]) -> Tuple[Dict[str, float], Optional[Tensor]]:
        known, unknown, condvec, random_condvec, noise1, noise2 = batch
        # mean = torch.zeros(known.shape[0], self._embedding_dim, device=self._device)
        # std = mean + 1
        enable_autocast = torch.cuda.is_available() and self._autocast
        # condvec = unknown[:, self._condvec_left:self._condvec_right]
        # if self._condvec_dim > 0:
        #     conditions = condvec.argmax(dim=1)
        #     for v in conditions:
        #         self._condvec_accumulated[v.item()] += 1
        known = self._make_context(known)

        for ds in range(self._discriminator_step):
            with torch.cuda.amp.autocast(enabled=enable_autocast):
                # fake_cat = self._construct_fake(mean, std, known)
                fake_cat = self._construct_fake(condvec, noise1, known)
                real_cat = torch.cat([known, condvec, unknown], dim=1)
                y_fake, y_real = self._discriminator(fake_cat), self._discriminator(real_cat)
                pen = self._discriminator.calc_gradient_penalty(
                    real_cat, fake_cat, self._device, self._pac
                ) #/ (known.shape[1] + self._embedding_dim)
                loss_d = -(torch.mean(y_real) - torch.mean(y_fake))
                if self._grad_scaler_d is None:
                    pen.backward(retain_graph=True)
                else:
                    self._grad_scaler_d.scale(pen).backward(retain_graph=True)
            self._take_step(loss_d, self._optimizer_d, self._grad_scaler_d, self._lr_schd_d)
            # TODO: for param in model.parameters(): param.grad = None

        with torch.cuda.amp.autocast(enabled=enable_autocast):
            # fake_cat = self._construct_fake(mean, std, known)
            if known.shape[1] == 0:
                fake_cat = self._construct_fake(random_condvec, noise2, known)
            else:
                fake_cat = self._construct_fake(condvec, noise2, known)
            real_cat = torch.cat([known, condvec, unknown], dim=1)
            y_fake = self._discriminator(fake_cat)
            if known.shape[1] == 0:
                distance = torch.tensor(0)
            else:
                distance = F.mse_loss(fake_cat[:, -self._unknown_dim:], real_cat[:, -self._unknown_dim:],
                                      reduction='mean')
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

    def _construct_fake(self, condvec: Tensor, noise: Tensor, known: Tensor) -> Tensor:
        fake = self._generator(torch.cat([noise, condvec, known], dim=1))
        fakeact = self._apply_activate(fake)
        fake_cat = torch.cat([known, condvec, fakeact], dim=1)
        return fake_cat

    # def _construct_fake(self, mean: Tensor, std: Tensor, known_tensor: Tensor) -> Tensor:
    #     fakez = torch.normal(mean=mean, std=std)[:known_tensor.shape[0]].to(self._device)
    #     if self._condvec_dim > 0:
    #         sum_cnt = sum(self._condvec_accumulated)
    #         probabilities = [x / sum_cnt for x in self._condvec_accumulated]
    #         conditions = np.random.choice(range(self._condvec_dim), known_tensor.shape[0], p=probabilities)
    #         condvec = F.one_hot(torch.from_numpy(conditions).long(), self._condvec_dim).to(self._device)
    #     else:
    #         condvec = known_tensor[:, 0:0]
    #     fakez = torch.cat([fakez, condvec, known_tensor], dim=1)
    #     fake = self._generator(fakez)
    #     fakeact = self._apply_activate(fake)
    #     fake_cat = torch.cat([known_tensor, condvec, fakeact], dim=1)
    #     return fake_cat

    @classmethod
    def _reconstruct(cls, distributed: bool, autocast: bool, log_dir: str, ckpt_dir: str, descr: str,
                     known_dim: int, unknown_dim: int, cat_dims: List[Tuple[int, int]], lae_trained: bool,
                     lae: nn.Module, optimizer_lae: Optimizer, lr_schd_lae: LRScheduler,
                     grad_scaler_lae: Optional[GradScaler],
                     # condvec_left: int, condvec_right: int, condvec_dim: int, condvec_accumulated: List[int],
                     cond_dist: List[Tensor], cond_span: List[Tuple[int, int]],
                     generator: nn.Module, optimizer_g: Optimizer, lr_schd_g: LRScheduler,
                     grad_scaler_g: Optional[GradScaler], discriminator: nn.Module, optimizer_d: Optimizer,
                     lr_schd_d: LRScheduler, grad_scaler_d: Optional[GradScaler],
                     embedding_dim: int, pac: int, discriminator_step: int, encoded_dim: int) -> "CTGANTrainer":
        base = TabularTrainer._reconstruct(
            distributed, autocast, log_dir, ckpt_dir, descr,
            known_dim, unknown_dim, cat_dims, lae_trained,
            lae, optimizer_lae, lr_schd_lae, grad_scaler_lae
        )
        base.__class__ = CTGANTrainer
        base._cond_dist, base._cond_span = cond_dist, cond_span
        # base._condvec_left, base._condvec_right, base._condvec_dim, base._condvec_accumulated = (
        #     condvec_left, condvec_right, condvec_dim, condvec_accumulated
        # )
        base._generator, base._optimizer_g, base._lr_schd_g, base._grad_scaler_g = (
            generator, optimizer_g, lr_schd_g, grad_scaler_g)
        base._discriminator, base._optimizer_d, base._lr_schd_d, base._grad_scaler_d = (
            discriminator, optimizer_d, lr_schd_d, grad_scaler_d)
        base._embedding_dim, base._pac, base._discriminator_step = embedding_dim, pac, discriminator_step
        base._encoded_dim = encoded_dim
        return base

    def __reduce__(self):
        _, var = super().__reduce__()
        return self._reconstruct, var + (
            self._cond_dist, self._cond_span,
            # self._condvec_left, self._condvec_right, self._condvec_dim, self._condvec_accumulated,
            self._generator, self._optimizer_g, self._lr_schd_g, self._grad_scaler_g,
            self._discriminator, self._optimizer_d, self._lr_schd_d, self._grad_scaler_d,
            self._embedding_dim, self._pac, self._discriminator_step, self._encoded_dim
        )

    def _apply_activate(self, data: Tensor) -> Tensor:
        act_data, ptr, cat_ptr = [], 0, 0
        while ptr < self._unknown_dim:
            if cat_ptr < len(self._cat_dims) and ptr == self._cat_dims[cat_ptr][0]:
                l, r = self._cat_dims[cat_ptr]
                cat_ptr += 1
                if r - l == 1:
                    act_data.append(torch.sigmoid(data[:, l:r]))
                else:
                    act_data.append(self._gumbel_softmax(data[:, l:r], tau=0.2))
                ptr = r
            else:
                act_data.append(torch.tanh(data[:, ptr:ptr+1]))
                ptr += 1
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
        # mean = torch.zeros(known.shape[0], self._embedding_dim, device=self._device)
        # std = mean + 1

        dataloader = self._make_dataloader(known, torch.zeros(known.shape[0], self._unknown_dim), batch_size, False)
        if is_main_process():
            dataloader = tqdm(dataloader)
            dataloader.set_description(f'Inference on {self._descr}')

        fakes, y_fakes = [], []
        self._generator.eval()
        self._discriminator.eval()
        for step, (known_batch, _, _, condvec, noise, _) in enumerate(dataloader):
            known_batch = known_batch.to(self._device)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available() and self._autocast):
                known = self._make_context(known_batch)
                fake_cat = self._construct_fake(condvec, noise, known)
                # fake_cat = self._construct_fake(mean, std, known)
                y_fake = self._discriminator(fake_cat)
                y_fake = y_fake.repeat(self._pac, 1).permute(1, 0).flatten()[:fake_cat.shape[0]]
                fakes.append(fake_cat)
                y_fakes.append(y_fake)
        self._generator.train()
        self._discriminator.train()

        return CTGANOutput(torch.cat(fakes), torch.cat(y_fakes))
