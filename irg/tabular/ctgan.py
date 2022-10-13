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
from ..utils.torch import Discriminator, LinearAutoEncoder


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
            - Generator arguments, all prefixed with "gen_" (for example, argument "arg1" under this group will be
              named as "gen_arg1").
                - `optimizer` (`str`): Optimizer type, currently support "SGD", "Adam", and "AdamW" only.
                  Default is "AdamW".
                - `scheduler` (`str`): LR scheduler type, currently support "StepLR" and "ConstantLR" only.
                  Default is "StepLR".
                - Optimizer constructor arguments, all prefixed with "optim_". (That is, argument "arg1" under this
                  group will be named as "gen_optim_arg1".
                - Scheduler constructor arguments, all prefixed with "sched_".
                - GradScaler constructor arguments, all prefixed with "scaler_".
            - Discriminator arguments, all prefixed with "disc_". Inner structure (except for the prefix) is the same
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

        # self._generator = Generator(
        #     embedding_dim=embedding_dim + self._known_dim + self._condvec_dim,
        #     generator_dim=generator_dim,
        #     data_dim=self._unknown_dim
        # ).to(self._device)
        # self._discriminator = Discriminator(
        #     input_dim=self._known_dim + self._unknown_dim + self._condvec_dim,
        #     discriminator_dim=discriminator_dim,
        #     pac=pac
        # ).to(self._device)
        encoded_dim = math.ceil(self._known_dim + self._unknown_dim / 10) # TODO ratio
        self._generator = Generator(
            embedding_dim=embedding_dim + encoded_dim + self._condvec_dim,
            generator_dim=generator_dim,
            data_dim=encoded_dim
        ).to(self._device)
        self._discriminator = Discriminator(
            input_dim=encoded_dim + self._condvec_dim,
            discriminator_dim=discriminator_dim,
            pac=pac
        ).to(self._device)
        self._lae = LinearAutoEncoder(
            full_dim=self._known_dim + self._unknown_dim,
            encoded_dim=encoded_dim
        ).to(self._device)
        self._generator, self._optimizer_g, self._lr_schd_g, self._grad_scaler_g = self._make_model_optimizer(
            self._generator,
            **{n[4:]: v for n, v in kwargs.items() if n.startswith('gen_')}
        )
        self._discriminator, self._optimizer_d, self._lr_schd_d, self._grad_scaler_d = self._make_model_optimizer(
            self._discriminator,
            **{n[5:]: v for n, v in kwargs.items() if n.startswith('disc_')}
        )
        self._lae, self._optimizer_l, self._lr_schd_l, self._grad_scaler_l = self._make_model_optimizer(
            self._lae # TODO args
        )

        self._embedding_dim, self._pac, self._discriminator_step = embedding_dim, pac, discriminator_step

    @property
    def _aux_info_path(self) -> str:
        return os.path.join(self._ckpt_dir, self._descr, 'info.pt')

    def _reload_checkpoint(self, idx: int, by: str):
        path = os.path.join(self._ckpt_dir, self._descr, f'{by}_{idx:07d}.pt')
        if not os.path.exists(path):
            return
        loaded = torch.load(path)
        self._load_content_from(loaded)

    def _load_content_from(self, loaded: Dict[str, Any]):
        generator_dict = loaded['generator']['model']
        if hasattr(self._generator, 'module'):
            generator_dict = OrderedDict({f'module.{n}': v for n, v in generator_dict.items()})
        self._generator.load_state_dict(generator_dict, strict=True)
        discriminator_dict = loaded['discriminator']['model']
        if hasattr(self._discriminator, 'module'):
            discriminator_dict = OrderedDict({f'module.{n}': v for n, v in discriminator_dict.items()})
        self._discriminator.load_state_dict(discriminator_dict, strict=True)
        lae_dict = loaded['lae']['model']
        if hasattr(self._discriminator, 'module'):
            lae_dict = OrderedDict({f'module.{n}': v for n, v in lae_dict.items()})
        self._discriminator.load_state_dict(discriminator_dict, strict=True)
        self._optimizer_g.load_state_dict(loaded['generator']['optimizer'])
        self._optimizer_d.load_state_dict(loaded['discriminator']['optimizer'])
        self._optimizer_l.load_state_dict(loaded['lae']['optimizer'])
        self._lr_schd_g.load_state_dict(loaded['generator']['lr_scheduler'])
        self._lr_schd_d.load_state_dict(loaded['discriminator']['lr_scheduler'])
        self._lr_schd_l.load_state_dict(loaded['lae']['lr_scheduler'])
        if 'grad_scaler' in loaded['generator']:
            self._grad_scaler_g.load_state_dict(loaded['generator']['grad_scaler'])
        else:
            self._grad_scaler_g = None
        if 'grad_scaler' in loaded['discriminator']:
            self._grad_scaler_d.load_state_dict(loaded['discriminator']['grad_scaler'])
        else:
            self._grad_scaler_d = None
        if 'grad_scaler' in loaded['lae']:
            self._grad_scaler_l.load_state_dict(loaded['lae']['grad_scaler'])
        else:
            self._grad_scaler_l = None
        self._condvec_accumulated, self._condvec_left, self._condvec_right, self._condvec_dim = loaded['condvec']
        torch.manual_seed(loaded['seed'])

    def _save_checkpoint(self, idx: int, by: str):
        torch.save(
            self._construct_content_to_save(),
            os.path.join(self._ckpt_dir, self._descr, f'{by}_{idx:07d}.pt')
        )

    def _construct_content_to_save(self) -> Dict[str, Any]:
        return {
            'generator': {
                'model': (self._generator.module if hasattr(self._generator, 'module')
                          else self._generator).state_dict(),
                'optimizer': self._optimizer_g.state_dict(),
                'lr_scheduler': self._lr_schd_g.state_dict()
            } | ({'grad_scaler': self._grad_scaler_g.state_dict()} if self._grad_scaler_g is not None else {}),
            'discriminator': {
                'model': (self._discriminator.module if hasattr(self._discriminator, 'module')
                          else self._discriminator).state_dict(),
                'optimizer': self._optimizer_d.state_dict(),
                'lr_scheduler': self._lr_schd_d.state_dict()
            } | ({'grad_scaler': self._grad_scaler_d.state_dict()} if self._grad_scaler_d is not None else {}),
            'lae': {
                'model': (self._lae.module if hasattr(self._lae, 'module')
                          else self._lae).state_dict(),
                'optimizer': self._optimizer_l.state_dict(),
                'lr_scheduler': self._lr_schd_l.state_dict()
            } | ({'grad_scaler': self._grad_scaler_l.state_dict()} if self._grad_scaler_l is not None else {}),
            'seed': torch.initial_seed(),
            'condvec': (self._condvec_accumulated, self._condvec_left, self._condvec_right, self._condvec_dim)
        }

    def run_step(self, known: Tensor, unknown: Tensor) -> Tuple[Dict[str, float], Optional[Tensor]]:
        mean = torch.zeros(known.shape[0], self._embedding_dim, device=self._device)
        std = mean + 1
        enable_autocast = torch.cuda.is_available() and self._autocast
        condvec = unknown[:, self._condvec_left:self._condvec_right]
        if self._condvec_dim > 0:
            conditions = condvec.argmax(dim=1)
            for v in conditions:
                self._condvec_accumulated[v.item()] += 1

        real_data = torch.cat([known, unknown], dim=1)
        reconstructed = self._lae(real_data, 'recon')
        reconstructed = self._apply_activate(reconstructed)
        recon_loss = F.mse_loss(reconstructed, real_data)
        self._take_step(recon_loss, self._optimizer_l, self._grad_scaler_l, self._lr_schd_l, True)
        self._lae.eval()
        with torch.no_grad():
            full_encoded = self._lae(real_data, 'enc')
            context_encoded = self._lae(torch.cat([known, torch.zeros(*unknown.shape).to(self._device)], dim=1), 'enc')

        for ds in range(self._discriminator_step):
            with torch.cuda.amp.autocast(enabled=enable_autocast):
                # fake_cat = self._construct_fake(mean, std, known)
                # real_cat = torch.cat([known, condvec, unknown], dim=1)
                fake_cat = self._construct_fake(mean, std, known, context_encoded)
                real_cat = torch.cat([condvec, full_encoded], dim=1)
                y_fake, y_real = self._discriminator(fake_cat), self._discriminator(real_cat)
                pen = self._discriminator.calc_gradient_penalty(
                    real_cat, fake_cat, self._device, self._pac
                ) / (known.shape[1] + self._embedding_dim)
                # pen = torch.sigmoid(pen)
                loss_d = -(torch.mean(y_real) - torch.mean(y_fake))
                if self._grad_scaler_d is None:
                    pen.backward(retain_graph=True)
                else:
                    self._grad_scaler_d.scale(pen).backward(retain_graph=True)
            self._take_step(loss_d, self._optimizer_d, self._grad_scaler_d, self._lr_schd_d)
            # TODO: for param in model.parameters(): param.grad = None

        with torch.cuda.amp.autocast(enabled=enable_autocast):
            # fake_cat = self._construct_fake(mean, std, known)
            fake_cat = self._construct_fake(mean, std, known, context_encoded)
            # real_cat = torch.cat([known, condvec, unknown], dim=1)
            real_cat = torch.cat([condvec, full_encoded], dim=1)
            y_fake = self._discriminator(fake_cat)
            if known.shape[1] == 0:
                distance = torch.tensor(0)
            else:
                distance = F.mse_loss(fake_cat, real_cat, reduction='mean')
            loss_g = -torch.mean(y_fake) + distance
        self._take_step(loss_g, self._optimizer_g, self._grad_scaler_g, self._lr_schd_g)
        self._lae.train()
        return {
                   'G loss': loss_g.detach().cpu().item(),
                   'distance': distance.detach().cpu().item(),
                   'D loss': loss_d.detach().cpu().item(),
                   'penalty': pen.detach().cpu().item(),
                   'G lr': self._optimizer_g.param_groups[0]['lr'],
                   'D lr': self._optimizer_d.param_groups[0]['lr']
               }, fake_cat

    # def _construct_fake(self, mean: Tensor, std: Tensor, known_tensor: Tensor) -> Tensor:
    def _construct_fake(self, mean: Tensor, std: Tensor, known_tensor: Tensor, context_encoded: Tensor) -> Tensor:
        fakez = torch.normal(mean=mean, std=std)[:known_tensor.shape[0]].to(self._device)
        if self._condvec_dim > 0:
            sum_cnt = sum(self._condvec_accumulated)
            probabilities = [x / sum_cnt for x in self._condvec_accumulated]
            conditions = np.random.choice(range(self._condvec_dim), known_tensor.shape[0], p=probabilities)
            condvec = F.one_hot(torch.from_numpy(conditions).long(), self._condvec_dim).to(self._device)
        else:
            condvec = known_tensor[:, 0:0]
        fakez = torch.cat([fakez, condvec, context_encoded], dim=1)
        fake = self._generator(fakez)
        fakeact = F.sigmoid(fake)
        # fakeact = self._apply_activate(fake)
        # fake_cat = torch.cat([known_tensor, condvec, fakeact], dim=1)
        # fake_cat = self._lae(torch.cat([known_tensor, fakeact], dim=1), 'enc')
        fake_cat = torch.cat([condvec, fakeact], dim=1)
        return fake_cat

    @classmethod
    def _reconstruct(cls, distributed: bool, autocast: bool, log_dir: str, ckpt_dir: str, descr: str,
                     cat_dims: List[Tuple[int, int]], known_dim: int, unknown_dim: int,
                     fitted_mean: Tensor, fitted_std: Tensor, total_train: int,
                     condvec_left: int, condvec_right: int, condvec_dim: int, condvec_accumulated: List[int],
                     generator: nn.Module, optimizer_g: Optimizer, lr_schd_g: LRScheduler,
                     grad_scaler_g: Optional[GradScaler], discriminator: nn.Module, optimizer_d: Optimizer,
                     lr_schd_d: LRScheduler, grad_scaler_d: Optional[GradScaler], lae: nn.Module,
                     optimizer_l: Optimizer, lr_schd_l: LRScheduler, grad_scaler_l: Optional[GradScaler],
                     embedding_dim: int, pac: int, discriminator_step: int) -> "CTGANTrainer":
        base = TabularTrainer._reconstruct(
            distributed, autocast, log_dir, ckpt_dir, descr,
            cat_dims, known_dim, unknown_dim, fitted_mean, fitted_std, total_train
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
        return base

    def __reduce__(self):
        _, var = super().__reduce__()
        return self._reconstruct, var + (
            self._condvec_left, self._condvec_right, self._condvec_dim, self._condvec_accumulated,
            self._generator, self._optimizer_g, self._lr_schd_g, self._grad_scaler_g,
            self._discriminator, self._optimizer_d, self._lr_schd_d, self._grad_scaler_d,
            self._lae, self._optimizer_l, self._lr_schd_l, self._grad_scaler_l,
            self._embedding_dim, self._pac, self._discriminator_step
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
        # mask = torch.cat(act_data, dim=1)
        # noise = self._make_noise(activated)
        # activated = activated + noise #* mask
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
        self._lae.eval()
        for step, (known_batch, _) in enumerate(dataloader):
            known_batch = known_batch.to(self._device)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available() and self._autocast):
                context_encoded = self._lae(torch.cat([
                    known_batch, torch.zeros(known_batch.shape[0], self.unknown_dim)
                ], dim=1), 'enc')
                fake_cat = self._construct_fake(mean, std, known_batch, context_encoded)
                y_fake = self._discriminator(fake_cat)
                y_fake = y_fake.repeat(self._pac, 1).permute(1, 0).flatten()[:fake_cat.shape[0]]
                fake_cat = self._lae(fake_cat)
                fake_cat = self._apply_activate(fake_cat)
                fakes.append(fake_cat)
                y_fakes.append(y_fake)
        self._generator.train()
        self._discriminator.train()

        return CTGANOutput(torch.cat(fakes), torch.cat(y_fakes))
