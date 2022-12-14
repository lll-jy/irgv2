"""Partial TVAE Training."""
from collections import OrderedDict
from typing import Tuple, Dict, Optional
import os

import torch
from torch import Tensor
from torch.nn import functional as F
from ctgan.synthesizers.tvae import Encoder, Decoder
from tqdm import tqdm

from .base import TabularTrainer
from ..utils import InferenceOutput
from ..utils.dist import is_main_process


class TVAEOutput(InferenceOutput):
    """Output of TVAE."""
    def __init__(self, rec: Tensor, sigmas: Tensor):
        super().__init__(rec)
        self.rec = rec
        """Reconstructed result."""
        self.sigmas = sigmas
        """Sigmas of reconstructed result."""


class TVAETrainer(TabularTrainer):
    """Trainer for TVAE."""
    def __init__(self, embedding_dim: int = 128, compress_dims: Tuple[int, ...] = (128, 128),
                 decompress_dims: Tuple[int, ...] = (128, 128), loss_factor: float = 2, **kwargs):
        """
        **Args**:

        - `embedding_dim` to `loss_factor`: Arguments for
          [TVAE](https://sdv.dev/SDV/api_reference/tabular/api/sdv.tabular.ctgan.TVAE.html#sdv.tabular.ctgan.TVAE)
        - `kwargs`: It has the following groups:
            - Inherited arguments from [`TabularTrainer`](./base#irg.tabular.base.TabularTrainer).
            - Model arguments, same as generator arguments without prefix for [CTGAN](.#irg.tabular.CTGANTrainer).
        """
        super().__init__(**{
            n: v for n, v in kwargs.items() if
            n in {'distributed', 'autocast', 'log_dir', 'ckpt_dir', 'descr',
                  'cat_dims', 'known_dim', 'unknown_dim'}
        })
        self._encoder = Encoder(
            data_dim=self._known_dim + self._unknown_dim,
            compress_dims=compress_dims,
            embedding_dim=embedding_dim
        ).to(self._device)
        self._decoder = Decoder(
            embedding_dim=embedding_dim + self._known_dim,
            decompress_dims=decompress_dims,
            data_dim=self._unknown_dim
        ).to(self._device)
        (self._encoder, self._decoder), self._optimizer, self._lr_schd, self._grad_scaler = self._make_model_optimizer(
            [self._encoder, self._decoder],
            **{n: v for n, v in kwargs.items() if
               n not in {'distributed', 'autocast', 'log_dir', 'ckpt_dir', 'descr',
                         'cat_dims', 'known_dim', 'unknown_dim'}}
        )

        self._loss_factor = loss_factor

    def _reload_checkpoint(self, idx: int, by: str):
        path = os.path.join(self._ckpt_dir, self._descr, f'{by}_{idx:07d}.pt')
        if not os.path.exists(path):
            return
        loaded = torch.load(path)
        encoder_dict = loaded['encoder']
        if hasattr(self._encoder, 'module'):
            encoder_dict = OrderedDict({f'module.{n}': v for n, v in encoder_dict.items()})
        self._encoder.load_state_dict(encoder_dict, strict=True)
        decoder_dict = loaded['decoder']
        if hasattr(self._decoder, 'module'):
            decoder_dict = OrderedDict({f'module.{n}': v for n, v in decoder_dict.items()})
        self._decoder.load_state_dict(decoder_dict, strict=True)
        self._optimizer.load_state_dict(loaded['optimizer'])
        self._lr_schd.load_state_dict(loaded['lr_scheduler'])
        self._grad_scaler.load_state_dict(loaded['grad_scaler'])
        torch.manual_seed(loaded['seed'])

    def _save_checkpoint(self, idx: int, by: str):
        torch.save(
            {
                'encoder': (self._encoder.module if hasattr(self._encoder, 'module') else self._encoder).state_dict(),
                'decoder': (self._decoder.module if hasattr(self._decoder, 'module') else self._decoder).state_dict(),
                'optimizer': self._optimizer.state_dict(),
                'lr_scheduler': self._lr_schd.state_dict(),
                'grad_scaler': self._grad_scaler.state_dict(),
                'seed': torch.initial_seed()
            },
            os.path.join(self._ckpt_dir, self._descr, f'{by}_{idx:07d}.pt')
        )

    def run_step(self, known: Tensor, unknown: Tensor) -> Tuple[Dict[str, float], Optional[Tensor]]:
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available() and self._autocast):
            real = torch.cat([known, unknown], dim=1)
            mu, std, logvar = self._encoder(real)
            eps = torch.randn_like(std)

            emb = eps * std + mu
            known_real = real[:, :self._known_dim]
            dec_input = torch.cat([emb, known_real], dim=1)
            rec, sigmas = self._decoder(dec_input)

            dist_loss, kld_loss = self._calc_loss(rec, real, sigmas, mu, logvar)
            loss = dist_loss + kld_loss

        self._take_step(loss, self._optimizer, self._grad_scaler, self._lr_schd)
        return {
            'Dist loss': dist_loss.detach().cpu().item(),
            'KLD loss': kld_loss.detach().cpu().item()
        }, rec

    def _calc_loss(self, rec: Tensor, real: Tensor, sigmas: Tensor, mu: Tensor, logvar: Tensor) \
            -> Tuple[Tensor, Tensor]:
        losses, ptr, cat_ptr = [], 0, 0
        while ptr < self._unknown_dim:
            if cat_ptr < len(self._cat_dims) and ptr == self._cat_dims[cat_ptr][0]:
                l, r = self._cat_dims[cat_ptr]
                cat_ptr += 1
                if r - l == 1:
                    losses.append(F.binary_cross_entropy(
                        rec[:, l:r].clamp(0, 1), real[:, l:r].clamp(0, 1), reduction='sum'))
                else:
                    losses.append(F.cross_entropy(
                        rec[:, l:r], torch.argmax(real[:, l:r], dim=-1), reduction='sum'))
                ptr = r
            else:
                std = sigmas[ptr]
                eq = real[:, ptr] - torch.tanh(rec[:, ptr])
                losses.append((eq ** 2 / 2 / (std ** 2)).sum())
                losses.append(torch.log(std) * real.size()[0])
                ptr += 1

        kld = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
        return sum(losses) * self._loss_factor / real.size()[0], kld / real.size()[0]

    @torch.no_grad()
    def inference(self, known: Tensor, batch_size: int) -> TVAEOutput:
        dataloader = self._make_dataloader(known, torch.rand(known.shape[0], self._unknown_dim), batch_size, False)
        if is_main_process():
            dataloader = tqdm(dataloader)
            dataloader.set_description(f'Inference on {self._descr}')

        rec, sigmas = [], []
        self._encoder.eval()
        self._decoder.eval()
        for step, (known_batch, unknown_batch) in enumerate(dataloader):
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available() and self._autocast):
                real = torch.cat([known_batch, unknown_batch], dim=1)
                mu, std, logvar = self._encoder(real)
                eps = torch.randn_like(std)

                emb = eps * std + mu
                known_real = real[:, :self._known_dim]
                dec_input = torch.cat([emb, known_real], dim=1)
                rec_batch, sigmas_batch = self._decoder(dec_input)
                rec.append(rec_batch)
                sigmas.append(sigmas_batch)
        self._encoder.train()
        self._decoder.train()

        return TVAEOutput(torch.stack(rec), torch.stack(sigmas))
