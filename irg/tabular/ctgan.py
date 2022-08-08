from collections import OrderedDict
from typing import Tuple, Dict, Optional, List
import os

import torch
from torch import Tensor
import torch.nn.functional as F
from packaging import version
from ctgan.synthesizers.ctgan import Generator, Discriminator

from ..utils import Trainer
from ..utils.dist import get_device


class CTGANTrainer(Trainer):
    def __init__(self, cat_dims: List[Tuple[int, int]], known_dim: int, unknown_dim: int, embedding_dim: int = 128,
                 generator_dim: Tuple[int] = (256, 256), discriminator_dim: Tuple[int] = (256, 256), pac: int = 10,
                 discriminator_step: int = 1, **kwargs):
        super().__init__(**{
            n: v for n, v in kwargs.items() if
            n in {'distributed', 'autocast', 'log_dir', 'ckpt_dir', 'descr'}
        })
        self._device = get_device()
        self._generator = Generator(embedding_dim+known_dim, generator_dim, unknown_dim).to(self._device)
        self._discriminator = Discriminator(known_dim+unknown_dim, discriminator_dim, pac=pac).to(self._device)
        self._generator, self._optimizer_g, self._lr_schd_g, self._grad_scaler_g = self._make_model_optimizer(
            self._generator,
            **{n[4:]: v for n, v in kwargs.items() if n.startswith('gen_')}
        )
        self._discriminator, self._optimizer_d, self._lr_schd_d, self._grad_scaler_d = self._make_model_optimizer(
            self._discriminator,
            **{n[5:]: v for n, v in kwargs.items() if n.startswith('disc_')}
        )

        self._known_dim, self._unknown_dim, self._embedding_dim, self._pac = known_dim, unknown_dim, embedding_dim, pac
        self._discriminator_step = discriminator_step
        self._cat_dims = sorted(cat_dims)
        if not self._validate_cat_dims(self._cat_dims):
            raise ValueError('Category dimensions should be disjoint.')

    @staticmethod
    def _validate_cat_dims(cat_dims) -> bool:
        pre = 0
        for l, r in cat_dims:
            if l < pre or r <= l:
                return False
            pre = r
        return True

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
        self._grad_scaler_g.load_state_dict(loaded['generator']['grad_scaler'])
        self._grad_scaler_d.load_state_dict(loaded['discriminator']['grad_scaler'])
        torch.manual_seed(loaded['seed'])

    def _save_checkpoint(self, idx: int, by: str):
        torch.save(
            {
                'generator': {
                    'model': (self._generator.module if hasattr(self._generator, 'module')
                              else self._generator).state_dict(),
                    'optimizer': self._optimizer_g.state_dict(),
                    'lr_scheduler': self._lr_schd_g.state_dict(),
                    'grad_scaler': self._grad_scaler_g.state_dict()
                },
                'discriminator': {
                    'model': (self._discriminator.module if hasattr(self._discriminator, 'module')
                              else self._discriminator).state_dict(),
                    'optimizer': self._optimizer_d.state_dict(),
                    'lr_scheduler': self._lr_schd_d.state_dict(),
                    'grad_scaler': self._grad_scaler_d.state_dict()
                },
                'seed': torch.initial_seed()
            },
            os.path.join(self._ckpt_dir, self._descr, f'{by}_{idx:07d}.pt')
        )

    def run_step(self, known: Tensor, unknown: Tensor) -> Tuple[Dict[str: float], Optional[Tensor]]:
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
                )
                loss_d = -(torch.mean(y_real) - torch.mean(y_fake))
                self._grad_scaler_d.scale(pen).backward(retain_graph=True)
            self._take_step(loss_d, self._optimizer_d, self._grad_scaler_d, self._lr_schd_d)
            # TODO: for param in model.parameters(): param.grad = None

        with torch.cuda.amp.autocast(enabled=enable_autocast):
            fake_cat = self._construct_fake(mean, std, known)
            real_cat = torch.cat([known, unknown], dim=1)
            y_fake = self._discriminator(fake_cat)
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
                transformed = F.gumbel_softmax(logits, tau=tau, hard=hard,
                                                        eps=eps, dim=dim)
                if not torch.isnan(transformed).any():
                    return transformed
            raise ValueError('gumbel_softmax returning NaN.')
        return F.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)
