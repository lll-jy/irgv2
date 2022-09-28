"""Partial CTGAN Training."""

from collections import OrderedDict
from typing import Tuple, Dict, Optional
import os

import torch
from torch import Tensor
import torch.nn.functional as F
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
    def __init__(self, embedding_dim: int = 128, generator_dim: Tuple[int, ...] = (256, 256),
                 discriminator_dim: Tuple[int, ...] = (256, 256),
                 pac: int = 10, discriminator_step: int = 1, **kwargs):
        """
        **Args**:

        - `embedding_dim` to `discriminator_step`: Arguments for
          [CTGAN](https://sdv.dev/SDV/api_reference/tabular/api/sdv.tabular.ctgan.CTGAN.html#sdv.tabular.ctgan.CTGAN).
        - `kwargs`: It has the following groups:
            - Inherited arguments from [`TabularTrainer`](./base#irg.tabular.base.TabularTrainer).
            - Generator arguments, all prefixed with "gen_" (for example, argument "arg1" under this group will be
              named as "gen_arg1").
                - `optimizer` (`str`): Optimizer type, currently support "SGD", "Adam", and "AdamW" only.
                  Default is "AdamW".
                - 'scheduler` (`str`): LR scheduler type, currently support "StepLR" and "ConstantLR" only.
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
        self._generator = Generator(
            embedding_dim=embedding_dim + self._known_dim,
            generator_dim=generator_dim,
            data_dim=self._unknown_dim
        ).to(self._device)
        self._discriminator = Discriminator(
            input_dim=self._known_dim + self._unknown_dim,
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

    def _reload_checkpoint(self, idx: int, by: str):
        path = os.path.join(self._ckpt_dir, self._descr, f'{by}_{idx:07d}.pt')
        if not os.path.exists(path):
            return
        loaded = torch.load(path)
        generator_dict = loaded['generator']['model']
        if hasattr(self._generator, 'module'):
            generator_dict = OrderedDict({f'module.{n}': v for n, v in generator_dict.items()})
        self._generator.load_state_dict(generator_dict, strict=True)
        discriminator_dict = loaded['discriminator']['model']
        if hasattr(self._discriminator, 'module'):
            discriminator_dict = OrderedDict({f'module.{n}': v for n, v in discriminator_dict.items()})
        self._discriminator.load_state_dict(discriminator_dict, strict=True)
        self._optimizer_g.load_state_dict(loaded['generator']['optimizer'])
        self._optimizer_d.load_state_dict(loaded['discriminator']['optimizer'])
        self._lr_schd_g.load_state_dict(loaded['generator']['lr_scheduler'])
        self._lr_schd_d.load_state_dict(loaded['discriminator']['lr_scheduler'])
        if 'grad_scaler' in loaded['generator']:
            self._grad_scaler_g.load_state_dict(loaded['generator']['grad_scaler'])
        else:
            self._grad_scaler_g = None
        if 'grad_scaler' in loaded['discriminator']:
            self._grad_scaler_d.load_state_dict(loaded['discrminator']['grad_scaler'])
        else:
            self._grad_scaler_d = None
        torch.manual_seed(loaded['seed'])

    def _save_checkpoint(self, idx: int, by: str):
        torch.save(
            {
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
                'seed': torch.initial_seed()
            },
            os.path.join(self._ckpt_dir, self._descr, f'{by}_{idx:07d}.pt')
        )

    def run_step(self, known: Tensor, unknown: Tensor) -> Tuple[Dict[str, float], Optional[Tensor]]:
        mean = torch.zeros(known.shape[0], self._embedding_dim, device=self._device)
        std = mean + 1
        enable_autocast = torch.cuda.is_available() and self._autocast
        for ds in range(self._discriminator_step):
            with torch.cuda.amp.autocast(enabled=enable_autocast):
                fake_cat = self._construct_fake(mean, std, known)
                real_cat = torch.cat([known, unknown], dim=1)
                print('before 3', self._ckpt_dir, self._descr, fake_cat.size(), real_cat.size())
                y_fake, y_real = self._discriminator(fake_cat), self._discriminator(real_cat)
                print('done 3')
                pen = self._discriminator.calc_gradient_penalty(
                    real_cat, fake_cat, self._device, self._pac
                )
                loss_d = -(torch.mean(y_real) - torch.mean(y_fake))
                if self._grad_scaler_d is None:
                    pen.backward(retain_graph=True)
                else:
                    self._grad_scaler_d.scale(pen).backward(retain_graph=True)
            self._take_step(loss_d, self._optimizer_d, self._grad_scaler_d, self._lr_schd_d)
            # TODO: for param in model.parameters(): param.grad = None

        with torch.cuda.amp.autocast(enabled=enable_autocast):
            fake_cat = self._construct_fake(mean, std, known)
            real_cat = torch.cat([known, unknown], dim=1)
            print('before discriminator', self._ckpt_dir, self._descr, fake_cat.size(), self._discriminator.pac)
            y_fake = self._discriminator(fake_cat)
            print('end discriminator')
            distance = F.mse_loss(fake_cat, real_cat, reduction='mean')
            loss_g = -torch.mean(y_fake) + distance
        self._take_step(loss_g, self._optimizer_g, self._grad_scaler_g, self._lr_schd_g)
        return {
            'G loss': loss_g.detach().cpu().item(),
            'D loss': loss_d.detach().cpu().item(),
            'penalty': pen.detach().cpu().item()
        }, fake_cat

    def _construct_fake(self, mean: Tensor, std: Tensor, known_tensor: Tensor) -> Tensor:
        fakez = torch.normal(mean=mean, std=std)
        fakez = torch.cat([fakez, known_tensor], dim=1)
        fake = self._generator(fakez)
        fakeact = self._apply_activate(fake)
        fake_cat = torch.cat([known_tensor, fakeact], dim=1)
        return fake_cat

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
        return torch.cat(act_data, dim=1)

    @staticmethod
    def _gumbel_softmax(logits: torch.Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1):
        if version.parse(torch.__version__) < version.parse('1.2.0'):
            for i in range(10):
                transformed = F.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)
                if not torch.isnan(transformed).any():
                    return transformed
            raise ValueError('gumbel_softmax returning NaN.')
        return F.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)

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
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available() and self._autocast):
                fake_cat = self._construct_fake(mean, std, known_batch)
                print('before 2', self._ckpt_dir, self._descr, fake_cat.size())
                y_fake = self._discriminator(fake_cat)
                print('done 2')
                fakes.append(fake_cat)
                y_fakes.append(y_fake)
        self._generator.train()
        self._discriminator.train()

        return CTGANOutput(torch.stack(fakes), torch.stack(y_fakes))
