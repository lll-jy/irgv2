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
from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import SpanInfo
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


# class CTGANTrainer(TabularTrainer):
#
#     def __init__(self, embedding_dim: int = 128,
#                  generator_dim: Tuple[int, ...] = (256, 256), discriminator_dim: Tuple[int, ...] = (256, 256),
#                  pac: int = 10, discriminator_step: int = 1, **kwargs):
#         """
#         **Args**:
#
#         - `embedding_dim` to `discriminator_step`: Arguments for
#           [CTGAN](https://sdv.dev/SDV/api_reference/tabular/api/sdv.tabular.ctgan.CTGAN.html#sdv.tabular.ctgan.CTGAN).
#         - `kwargs`: It has the following groups:
#             - Inherited arguments from [`TabularTrainer`](./base#irg.tabular.base.TabularTrainer).
#             - Generator arguments, all prefixed with "gen_", and others are the same as arguments for AutoEncoder for
#               the parent class [`TabularTrainer`](./base#irg.tabular.base.TabularTrainer).
#             - Discriminator arguments, all prefixed with "disc_".
#               as generator.
#         """
#         super().__init__(**{
#             n: v for n, v in kwargs.items() if
#             n in {'distributed', 'autocast', 'log_dir', 'ckpt_dir', 'descr',
#                   'cat_dims', 'known_dim', 'unknown_dim'}
#         })
#         self._has_known = self._known_dim != 0
#         self._cond_span, self._unknown_info_list, self._cond_dim = [], [], 0
#         self._learn_condvec()
#         self._encoded_dim = 0 if self._known_dim == 0 else self._lae.encoded_dim
#         self._sampler: Optional[DataSampler] = None
#
#         self._generator = Generator(
#             embedding_dim=embedding_dim + self._encoded_dim + self._cond_dim,
#             generator_dim=generator_dim,
#             data_dim=self._unknown_dim
#         ).to(self._device)
#         self._discriminator = Discriminator(
#             input_dim=self._encoded_dim + self._cond_dim + self._unknown_dim,
#             discriminator_dim=discriminator_dim,
#             pac=pac
#         )
#         self._generator, self._optimizer_g, self._lr_schd_g, self._grad_scaler_g = self._make_model_optimizer(
#             self._generator,
#             **{n[4:]: v for n, v in kwargs.items() if n.startswith('gen_')}
#         )
#         self._discriminator, self._optimizer_d, self._lr_schd_d, self._grad_scaler_d = self._make_model_optimizer(
#             self._discriminator,
#             **{n[5:]: v for n, v in kwargs.items() if n.startswith('disc_')}
#         )
#         if os.path.exists(self._aux_info_path):
#             self._sampler, self._cond_span, self._unknown_info_list, self._cond_dim = torch.load(self._aux_info_path)
#
#         self._embedding_dim, self._discriminator_step, self._pac = embedding_dim, discriminator_step, pac
#
#     @property
#     def _aux_info_path(self) -> str:
#         return os.path.join(self._ckpt_dir, self._descr, 'info.pt')
#
#     def _learn_condvec(self):
#         if self._has_known:
#             return
#         is_for_num, ptr, cat_ptr = False, 0, 0
#         while ptr < self._unknown_dim:
#             if cat_ptr < len(self._cat_dims) and ptr == self._cat_dims[cat_ptr][0]:
#                 l, r = self._cat_dims[cat_ptr]
#                 cat_ptr += 1
#                 if r - l > 1 and not is_for_num:
#                     self._cond_span.append((l, r))
#                     self._unknown_info_list.append([SpanInfo(r-l, 'softmax')])
#                     self._cond_dim += r - l
#                 elif r - l == 1:
#                     self._unknown_info_list.append([SpanInfo(1, 'sigmoid')])
#                 else:
#                     self._unknown_info_list.append([SpanInfo(1, 'tanh'), SpanInfo(r-l, 'softmax')])
#                 is_for_num = False
#                 ptr = r
#             else:
#                 ptr += 1
#                 is_for_num = True
#
#     def _load_content_from(self, loaded: Dict[str, Any]):
#         super()._load_content_from(loaded)
#         self._generator, self._optimizer_g, self._lr_schd_g, self._grad_scaler_g = self._load_state_dict(
#             self._generator, self._optimizer_g, self._lr_schd_g, self._grad_scaler_g, loaded['generator']
#         )
#         self._discriminator, self._optimizer_d, self._lr_schd_d, self._grad_scaler_d = self._load_state_dict(
#             self._discriminator, self._optimizer_d, self._lr_schd_d, self._grad_scaler_d, loaded['discriminator']
#         )
#
#     def _construct_content_to_save(self) -> Dict[str, Any]:
#         return {
#                    'generator': self._full_state_dict(
#                        self._generator, self._optimizer_g, self._lr_schd_g, self._grad_scaler_g
#                    ),
#                    'discriminator': self._full_state_dict(
#                        self._discriminator, self._optimizer_d, self._lr_schd_d, self._grad_scaler_d
#                    )
#                } | super()._construct_content_to_save()
#
#     def _make_dataloader(self, known: Tensor, unknown: Tensor, batch_size: int, shuffle: bool = True) -> DataLoader:
#         if self._has_known:
#             return super()._make_dataloader(known, unknown, batch_size, shuffle)
#         self._sampler = DataSampler(
#             data=unknown.numpy(),
#             output_info=self._unknown_info_list,
#             log_frequency=True
#         )
#         torch.save((self._sampler, self._cond_span, self._unknown_info_list, self._cond_dim), self._aux_info_path)
#         return super()._make_dataloader(known, unknown, batch_size, shuffle)
#
#     def _collate_fn(self, batch: List[Tuple[Tensor, ...]]):
#         batch_size = len(batch)
#         mean = torch.zeros(batch_size, self._embedding_dim)
#         std = mean + 1
#
#         disc_in = []
#         for i in range(self._discriminator_step):
#             fakez = torch.normal(mean=mean, std=std)
#             condvec = self._sampler.sample_condvec(batch_size)
#             if condvec is None:
#                 c1, m1, col, opt, c2 = [torch.zeros(batch_size, 0) for _ in range(5)]
#                 real = self._sampler.sample_data(batch_size, col, opt)
#             else:
#                 c1, m1, col, opt = condvec
#                 c1 = torch.from_numpy(c1)
#                 m1 = torch.from_numpy(m1)
#
#                 perm = np.arange(batch_size)
#                 np.random.shuffle(perm)
#                 real = self._sampler.sample_data(
#                     batch_size, col[perm], opt[perm])
#                 c2 = c1[perm]
#             real = torch.from_numpy(real.astype('float32'))
#             disc_in.append((fakez, real, c1, c2, m1, col, opt))
#
#         fakez = torch.normal(mean=mean, std=std)
#         condvec = self._sampler.sample_condvec(batch_size)
#
#         if condvec is None:
#             c1, m1, col, opt = [torch.zeros(batch_size, 0) for _ in range(4)]
#         else:
#             c1, m1, col, opt = condvec
#             c1 = torch.from_numpy(c1)
#             m1 = torch.from_numpy(m1)
#         gen_in = fakez, c1, m1
#         return disc_in, gen_in
#
#     def _apply_activate(self, unknown: Tensor):
#         data_t = []
#         st = 0
#         for column_info in self._unknown_info_list:
#             for span_info in column_info:
#                 ed = st + span_info.dim
#                 if span_info.activation_fn == 'tanh':
#                     data_t.append(torch.tanh(unknown[:, st:ed]))
#                 elif span_info.activation_fn == 'softmax':
#                     transformed = self._gumbel_softmax(unknown[:, st:ed], tau=0.2)
#                     data_t.append(transformed)
#                 elif span_info.activation_fn == 'sigmoid':
#                     data_t.append(F.sigmoid(unknown[:, st:ed]))
#                 else:
#                     raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')
#                 st = ed
#
#     @staticmethod
#     def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
#         if version.parse(torch.__version__) < version.parse('1.2.0'):
#             for i in range(10):
#                 transformed = F.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=-1)
#                 if not torch.isnan(transformed).any():
#                     return transformed
#             raise ValueError('gumbel_softmax returning NaN.')
#
#         return F.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=-1)
#
#     def run_step(self, batch: Tuple[Tensor, ...]) -> Tuple[Dict[str, float], Optional[Tensor]]:
#         enable_autocast = torch.cuda.is_available() and self._autocast
#         if not self._has_known:
#             disc_in, gen_in = batch
#             for fakez, real, c1, c2, m1, col, opt in disc_in:
#                 with torch.cuda.amp.autocast(enabled=enable_autocast):
#                     fake = self._generator(torch.cat([fakez, c1], dim=1))
#                     fakeact = self._apply_activate(fake)
#                     real_cat = torch.cat([real, c2], dim=1)
#                     fake_cat = torch.cat([fakeact, c1], dim=1)
#                     y_fake = self._discriminator(fake_cat)
#                     y_real = self._discriminator(real_cat)
#                     pen = self._discriminator.calc_gradient_penalty(
#                         real_cat, fake_cat, self._device, self._pac)
#                     loss_d = -(torch.mean(y_real) - torch.mean(y_fake))
#                 self._optimizer_d.zero_grad()
#                 if self._grad_scaler_d is None:
#                     pen.backward(retain_graph=True)
#                 else:
#                     self._grad_scaler_d.scale(pen).backward(retain_graph=True)
#                 self._take_step(loss_d, self._optimizer_d, self._grad_scaler_d, self._lr_schd_d, do_zero_grad=False)
#             with torch.cuda.amp.autocast(enabled=enable_autocast):
#                 fakez, c1, m1 = gen_in
#                 fake = self._generator(torch.cat([fakez, c1], dim=1))
#                 fakeact = self._apply_activate(fake)
#                 fake_cat = torch.cat([fakeact, c1], dim=1)
#                 y_fake = self._discriminator(fake_cat)
#                 cross_entropy = torch.tensor(0) if c1.shape[1] == 0 else self._cond_loss(fake, c1, m1)
#                 loss_g = -torch.mean(y_fake) + cross_entropy
#             self._take_step(loss_g, self._optimizer_g, self._grad_scaler_g, self._lr_schd_g)
#             return {
#                        'G loss': loss_g.detach().cpu().item(),
#                        'ce': cross_entropy.detach().cpu().item(),
#                        # 'meta': meta_loss.detach().cpu().item(),
#                        'D loss': loss_d.detach().cpu().item(),
#                        # 'distance': distance.detach().cpu().item(),
#                        'penalty': pen.detach().cpu().item(),
#                        'G lr': self._optimizer_g.param_groups[0]['lr'],
#                        'D lr': self._optimizer_d.param_groups[0]['lr']
#                    }, fake_cat
#         else:
#             known, unknown = batch
#             known = self._make_context(known)
#
#     def _cond_loss(self, data, c, m):
#         loss = []
#         st = 0
#         st_c = 0
#         for column_info in self._unknown_info_list:
#             for span_info in column_info:
#                 if len(column_info) != 1 or span_info.activation_fn != 'softmax':
#                     # not discrete column
#                     st += span_info.dim
#                 else:
#                     ed = st + span_info.dim
#                     ed_c = st_c + span_info.dim
#                     tmp = F.cross_entropy(
#                         data[:, st:ed],
#                         torch.argmax(c[:, st_c:ed_c], dim=1),
#                         reduction='none'
#                     )
#                     loss.append(tmp)
#                     st = ed
#                     st_c = ed_c
#
#         loss = torch.stack(loss, dim=1)  # noqa: PD013
#
#         return (loss * m).sum() / data.size()[0]
#
#     def inference(self, known: Tensor, batch_size: int) -> InferenceOutput:



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

        self._has_known = self._known_dim != 0
        self._cond_span = []
        self._learn_cond_span()

        self._encoded_dim = 0 if self._known_dim == 0 else self._lae.encoded_dim
        context_dim = self._encoded_dim + (sum(r-l for l, r in self._cond_span) if len(self._cond_span) > 0 else 0)
        self._data_sampler: Optional[DataSampler] = None
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
        self._unknown_info_list = []
        if os.path.exists(self._aux_info_path):
            self._data_sampler, self._unknown_info_list = torch.load(self._aux_info_path)

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
        self._cond_span, self._unknown_info_list, self._data_sampler = loaded['sampler']

    def _construct_content_to_save(self) -> Dict[str, Any]:
        return {
                   'generator': self._full_state_dict(
                       self._generator, self._optimizer_g, self._lr_schd_g, self._grad_scaler_g
                   ),
                   'discriminator': self._full_state_dict(
                       self._discriminator, self._optimizer_d, self._lr_schd_d, self._grad_scaler_d
                   ),
                   'sampler': (self._cond_span, self._unknown_info_list, self._data_sampler)
               } | super()._construct_content_to_save()

    def _learn_cond_span(self):
        if self._has_known:
            return
        is_for_num, ptr, cat_ptr = False, 0, 0
        while ptr < self._unknown_dim:
            if cat_ptr < len(self._cat_dims) and ptr == self._cat_dims[cat_ptr][0]:
                l, r = self._cat_dims[cat_ptr]
                cat_ptr += 1
                if r - l > 1 and not is_for_num:
                    self._cond_span.append((l, r))
                is_for_num = False
                ptr = r
            else:
                ptr += 1
                is_for_num = True

    def _make_dataloader(self, known: Tensor, unknown: Tensor, batch_size: int, shuffle: bool = True) -> DataLoader:
        unknown_info_list = []
        is_for_num, ptr, cat_ptr = False, 0, 0
        while ptr < self._unknown_dim:
            if cat_ptr < len(self._cat_dims) and ptr == self._cat_dims[cat_ptr][0]:
                l, r = self._cat_dims[cat_ptr]
                cat_ptr += 1
                if r - l > 1 and not is_for_num:
                    unknown_info_list.append([SpanInfo(r-l, 'softmax')])
                elif r - l == 1:
                    unknown_info_list.append([SpanInfo(1, 'sigmoid')])
                else:
                    unknown_info_list.append([SpanInfo(1, 'tanh'), SpanInfo(r-l, 'softmax')])
                is_for_num = False
                ptr = r
            else:
                ptr += 1
                is_for_num = True
        self._unknown_info_list = unknown_info_list

        self._data_sampler = DataSampler(
            data=torch.cat([known, unknown], dim=1).numpy(),
            output_info=[[SpanInfo(1, 'tanh')] for _ in range(known.shape[1])] + unknown_info_list,
            log_frequency=True
        )
        return super()._make_dataloader(known, unknown, batch_size, shuffle)

    def _collate_fn(self, batch: List[Tuple[Tensor, ...]]):
        batch_size = len(batch)
        mean = torch.zeros(batch_size, self._embedding_dim)
        std = mean + 1
        noise1 = torch.normal(mean=mean, std=std)
        noise2 = torch.normal(mean=mean, std=std)
        condvec = self._data_sampler.sample_condvec(batch_size)
        if condvec is None:
            c1, m1, col, opt, c2 = [torch.zeros(batch_size, 0) for _ in range(5)]
            real = self._data_sampler.sample_data(batch_size, col, opt)
        else:
            c1, m1, col, opt = condvec
            c1 = torch.from_numpy(c1)

            perm = np.arange(batch_size)
            np.random.shuffle(perm)
            real = self._data_sampler.sample_data(
                batch_size, col[perm], opt[perm])
            c2 = c1[perm]
        known, unknown = torch.from_numpy(real[:, :-self._unknown_dim]), torch.from_numpy(real[:, -self._unknown_dim:])
        disc_round_out = known, unknown, noise1, c1, c2

        condvec = self._data_sampler.sample_condvec(batch_size)

        if condvec is None:
            c1, m1, col, opt = [torch.zeros(batch_size, 0) for _ in range(4)]
        else:
            c1, m1, col, opt = condvec
            c1 = torch.from_numpy(c1)
            m1 = torch.from_numpy(m1)
        gen_round_out = noise2, c1, m1

        return disc_round_out + gen_round_out

    def run_step(self, batch: Tuple[Tensor, ...]) -> Tuple[Dict[str, float], Optional[Tensor]]:
        known, unknown, noise1, dc1, dc2, noise2, gc1, gm1 = batch
        enable_autocast = torch.cuda.is_available() and self._autocast
        known = self._make_context(known)

        for ds in range(self._discriminator_step):
            with torch.cuda.amp.autocast(enabled=enable_autocast):
                fake_cat = self._construct_fake(dc1, noise1, known)
                real_cat = torch.cat([known, dc2, unknown], dim=1)
                y_fake, y_real = self._discriminator(fake_cat), self._discriminator(real_cat)
                pen = self._discriminator.calc_gradient_penalty(
                    real_cat, fake_cat, self._device, self._pac
                ) #/ (known.shape[1] + self._embedding_dim)
                loss_d = -(torch.mean(y_real) - torch.mean(y_fake))
                self._optimizer_d.zero_grad()
                if self._grad_scaler_d is None:
                    pen.backward(retain_graph=True)
                else:
                    self._grad_scaler_d.scale(pen).backward(retain_graph=True)
            self._take_step(loss_d, self._optimizer_d, self._grad_scaler_d, self._lr_schd_d, do_zero_grad=False)
            # TODO: for param in model.parameters(): param.grad = None

        with torch.cuda.amp.autocast(enabled=enable_autocast):
            fake_cat = self._construct_fake(gc1, noise2, known)
            real_cat = torch.cat([known, gc1, unknown], dim=1)
            y_fake = self._discriminator(fake_cat)
            if known.shape[1] == 0:
                distance = torch.tensor(0)
            else:
                distance = F.mse_loss(fake_cat[:, -self._unknown_dim:], real_cat[:, -self._unknown_dim:],
                                      reduction='mean')
            if len(self._cond_span) == 0:
                ce = torch.tensor(0)
            else:
                ce = self._cond_loss(fake_cat[:, -self._unknown_dim:], gc1, gm1)
            meta_loss = self._meta_loss(known, unknown, fake_cat[:, -self.unknown_dim:])
            loss_g = -torch.mean(y_fake) + distance + meta_loss + ce
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

    def _cond_loss(self, unknown: Tensor, c: Tensor, m: Tensor) -> Tensor:
        loss = []
        st = 0
        st_c = 0
        for column_info in self._unknown_info_list:
            for span_info in column_info:
                if len(column_info) != 1 or span_info.activation_fn != 'softmax':
                    st += span_info.dim
                else:
                    ed = st + span_info.dim
                    ed_c = st_c + span_info.dim
                    tmp = F.cross_entropy(
                        unknown[:, st:ed],
                        torch.argmax(c[:, st_c:ed_c], dim=1),
                        reduction='none'
                    )
                    loss.append(tmp)
                    st = ed
                    st_c = ed_c

        loss = torch.stack(loss, dim=1)  # noqa: PD013

        return (loss * m).sum() / unknown.size()[0]

    def _construct_fake(self, condvec: Tensor, noise: Tensor, known: Tensor) -> Tensor:
        fake = self._generator(torch.cat([noise, condvec, known], dim=1))
        fakeact = self._apply_activate(fake)
        fake_cat = torch.cat([known, condvec, fakeact], dim=1)
        return fake_cat

    @classmethod
    def _reconstruct(cls, distributed: bool, autocast: bool, log_dir: str, ckpt_dir: str, descr: str,
                     known_dim: int, unknown_dim: int, cat_dims: List[Tuple[int, int]], lae_trained: bool,
                     lae: nn.Module, optimizer_lae: Optimizer, lr_schd_lae: LRScheduler,
                     grad_scaler_lae: Optional[GradScaler], cond_span: List[Tuple[int, int]],
                     data_sampler: Optional[DataSampler], has_known: bool,
                     unknown_info_list: List[List[SpanInfo]],
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
        base._data_sampler, base._cond_span = data_sampler, cond_span
        base._unknown_info_list = has_known, unknown_info_list
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
            self._cond_span, self._data_sampler, self._has_known, self._unknown_info_list,
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

    def _collate_fn_infer(self, batch: List[Tuple[Tensor, ...]]) -> Tuple[Tensor, ...]:
        batch_size = len(batch)
        mean = torch.zeros(batch_size, self._embedding_dim)
        std = mean + 1
        fakez = torch.normal(mean=mean, std=std)
        condvec = self._data_sampler.sample_original_condvec(batch_size)
        if condvec is None:
            c1 = torch.zeros(batch_size, 0)
        else:
            c1 = torch.from_numpy(condvec)
        return super()._collate_fn_infer(batch) + (fakez, c1)

    @torch.no_grad()
    def inference(self, known: Tensor, batch_size: int) -> CTGANOutput:
        # mean = torch.zeros(known.shape[0], self._embedding_dim, device=self._device)
        # std = mean + 1

        dataloader = super()._make_infer_dataloader(known, batch_size, False)
        if is_main_process():
            dataloader = tqdm(dataloader)
            dataloader.set_description(f'Inference on {self._descr}')

        fakes, y_fakes = [], []
        self._generator.eval()
        self._discriminator.eval()
        for step, (known_batch, noise, gc1) in enumerate(dataloader):
        # for step, (known_batch, _, _, _, _, noise, gc1, _) in enumerate(dataloader):
            known_batch, gc1, noise = known_batch.to(self._device), gc1.to(self._device), noise.to(self._device)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available() and self._autocast):
                known = self._make_context(known_batch)
                fake_cat = self._construct_fake(gc1, noise, known)
                y_fake = self._discriminator(fake_cat)
                y_fake = y_fake.repeat(self._pac, 1).permute(1, 0).flatten()[:fake_cat.shape[0]]
                fakes.append(fake_cat)
                y_fakes.append(y_fake)
        self._generator.train()
        self._discriminator.train()

        return CTGANOutput(torch.cat(fakes), torch.cat(y_fakes))
