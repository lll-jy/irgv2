"""Series generation using TimeGAN trainer.
Adapted from https://github.com/benearnthof/TimeGAN/blob/main/modules_and_training.py."""
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn, Tensor
from torch.cuda.amp import GradScaler
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base import SeriesTrainer
from ..utils.torch import TimeGanNet
from ..utils import dist, SeriesInferenceOutput
from ..utils.dist import to_device

_LOGGER = logging.getLogger()


class TimeGANOutput(SeriesInferenceOutput):
    def __init__(self, fake: Tensor, lengths: List[int], discr_out: Optional[Tensor] = None):
        super().__init__(fake, lengths)
        self.fake = fake
        """Fake data generated."""
        self.discr_out = discr_out
        """Discriminator output."""


class TimeGANTrainer(SeriesTrainer):
    """Trainer for series table generation for TimeGAN."""
    def __init__(self, hidden_dim: int = 40, n_layers: int = 1, gamma: float = 1., **kwargs):
        super().__init__(**kwargs)
        print('dim', self._known_dim, self._unknown_dim, flush=True)
        self._embedder = TimeGanNet(
            input_size=self._known_dim + self._unknown_dim + 1, output_size=hidden_dim,
            hidden_dim=hidden_dim, n_layers=n_layers, activation='sigmoid'
        ).to(self._device)
        self._recovery = TimeGanNet(
            input_size=hidden_dim, output_size=self._unknown_dim + 1, hidden_dim=hidden_dim, n_layers=n_layers
        ).to(self._device)
        self._generator = TimeGanNet(
            input_size=self._known_dim + self._unknown_dim + 1, output_size=hidden_dim,
            hidden_dim=hidden_dim, n_layers=n_layers, activation='sigmoid'
        ).to(self._device)
        self._supervisor = TimeGanNet(
            input_size=hidden_dim, output_size=hidden_dim, hidden_dim=hidden_dim, n_layers=n_layers,
            activation='sigmoid'
        ).to(self._device)
        self._discriminator = TimeGanNet(
            input_size=hidden_dim, output_size=1, hidden_dim=hidden_dim, n_layers=n_layers,
        ).to(self._device)

        self._embedder, self._opt_e, self._lrs_e, self._gs_e = self._make_model_optimizer(
            self._embedder, **kwargs
        )
        self._recovery, self._opt_r, self._lrs_r, self._gs_r = self._make_model_optimizer(
            self._recovery, **kwargs
        )
        self._generator, self._opt_g, self._lrs_g, self._gs_g = self._make_model_optimizer(
            self._generator, **kwargs
        )
        self._supervisor, self._opt_s, self._lrs_s, self._gs_s = self._make_model_optimizer(
            self._supervisor, **kwargs
        )
        self._discriminator, self._opt_d, self._lrs_d, self._gs_d = self._make_model_optimizer(
            self._discriminator, **kwargs
        )
        self._hidden_dim = hidden_dim
        self._pretrained = False, False
        self._gamma = gamma
        self._cat_dims.append((self._known_dim, self._known_dim + 1))

    def _load_content_from(self, loaded: Dict[str, Any]):
        super()._load_content_from(loaded)
        self._embedder, self._opt_e, self._lrs_e, self._gs_e = self._load_state_dict(
            self._embedder, self._opt_e, self._lrs_e, self._gs_e, loaded['embedder']
        )
        self._recovery, self._opt_r, self._lrs_r, self._gs_r = self._load_state_dict(
            self._recovery, self._opt_r, self._lrs_r, self._gs_r, loaded['recovery']
        )
        self._generator, self._opt_g, self._lrs_g, self._gs_g = self._load_state_dict(
            self._generator, self._opt_g, self._lrs_g, self._gs_g, loaded['generator']
        )
        self._supervisor, self._opt_s, self._lrs_s, self._gs_s = self._load_state_dict(
            self._supervisor, self._opt_s, self._lrs_s, self._gs_s, loaded['supervisor']
        )
        self._discriminator, self._opt_d, self._lrs_d, self._gs_d = self._load_state_dict(
            self._discriminator, self._opt_d, self._lrs_d, self._gs_d, loaded['discriminator']
        )
        self._pretrained = loaded['pretrained']

    def _construct_content_to_save(self) -> Dict[str, Any]:
        return {
            'embedder': self._full_state_dict(self._embedder, self._opt_e, self._lrs_e, self._gs_e),
            'recovery': self._full_state_dict(self._recovery, self._opt_r, self._lrs_r, self._gs_r),
            'generator': self._full_state_dict(self._generator, self._opt_g, self._lrs_g, self._gs_g),
            'supervisor': self._full_state_dict(self._supervisor, self._opt_s, self._lrs_s, self._gs_s),
            'discriminator': self._full_state_dict(self._discriminator, self._opt_d, self._lrs_d, self._gs_d),
            'pretrained': self._pretrained
        } | super()._construct_content_to_save()

    @classmethod
    def _reconstruct(cls, distributed: bool, autocast: bool, log_dir: str, ckpt_dir: str, descr: str,
                     known_dim: int, unknown_dim: int, cat_dims: List[Tuple[int, int]], lae_trained: bool,
                     lae: nn.Module, optimizer_lae: Optimizer, lr_schd_lae: LRScheduler,
                     grad_scaler_lae: Optional[GradScaler], corr_mat: Optional[Tensor],
                     corr_mask: Optional[Tensor], mean: Optional[Tensor],
                     base_ids: Tuple[List[int], List[int]], seq_ids: Tuple[List[int], List[int]],
                     embedder: TimeGanNet, opt_e: Optimizer, lrs_e: LRScheduler, gs_e: Optional[GradScaler],
                     recovery: TimeGanNet, opt_r: Optimizer, lrs_r: LRScheduler, gs_r: Optional[GradScaler],
                     generator: TimeGanNet, opt_g: Optimizer, lrs_g: LRScheduler, gs_g: Optional[GradScaler],
                     supervisor: TimeGanNet, opt_s: Optimizer, lrs_s: LRScheduler, gs_s: Optional[GradScaler],
                     discriminator: TimeGanNet, opt_d: Optimizer, lrs_d: LRScheduler, gs_d: Optional[GradScaler],
                     hidden_dim: int, pretrained: (bool, bool), gamma: float) ->\
            "TimeGANTrainer":
        base = SeriesTrainer._reconstruct(
            distributed, autocast, log_dir, ckpt_dir, descr,
            known_dim, unknown_dim, cat_dims, lae_trained,
            lae, optimizer_lae, lr_schd_lae, grad_scaler_lae,
            corr_mat, corr_mask, mean, base_ids, seq_ids
        )
        base.__class__ = TimeGANTrainer
        base._embedder, base._opt_e, base._lrs_e, base._gs_e = embedder, opt_e, lrs_e, gs_e
        base._recovery, base._opt_r, base._lrs_r, base._gs_r = recovery, opt_r, lrs_r, gs_r
        base._generator, base._opt_g, base._lrs_g, base._gs_g = generator, opt_g, lrs_g, gs_g
        base._supervisor, base._opt_s, base._lrs_s, base._gs_s = supervisor, opt_s, lrs_s, gs_s
        base._discriminator, base._opt_d, base._lrs_d, base._gs_d = discriminator, opt_d, lrs_d, gs_d
        base._hidden_dim, base._pretrained, base._gamma = hidden_dim, pretrained, gamma
        return base

    def __reduce__(self):
        _, var = super().__reduce__()
        return self._reconstruct, var + (
            self._embedder, self._opt_e, self._lrs_e, self._gs_e,
            self._recovery, self._opt_r, self._lrs_r, self._gs_r,
            self._generator, self._opt_g, self._lrs_g, self._gs_g,
            self._supervisor, self._opt_s, self._lrs_s, self._gs_s,
            self._discriminator, self._opt_d, self._lrs_d, self._gs_d,
            self._hidden_dim, self._pretrained, self._gamma
        )

    def train(self, known: List[Tensor], unknown: List[Tensor], epochs: int = 10, batch_size: int = 100, shuffle: bool = True,
              save_freq: int = 100, resume: bool = True, lae_epochs: int = 10):
        (epoch, global_step), dataloader = self._prepare_training(known, unknown, batch_size, shuffle, resume)
        self._run_embedding(dataloader, epochs)
        self._run_training(epoch, epochs, global_step, save_freq, dataloader)

    def _run_embedding(self, dataloader: DataLoader, epochs: int = 10):
        if self._pretrained[0]:
            return
        self._embedder.train()
        self._recovery.train()
        for i in range(epochs):
            descr = f'Epoch[{i}] {self._descr} Emb'
            if dist.is_main_process():
                dataloader = tqdm(dataloader)
                dataloader.set_description(descr)
            for step, (_, known_batch, unknown_batch) in enumerate(dataloader):
                data = torch.cat([known_batch, unknown_batch], dim=-1)
                h, x_tilde = self._recover(data)
                print('shape here', known_batch.shape, data.shape)
                e_loss = self._calculate_recon_loss(data[:, :, known_batch.shape[-1]:], x_tilde, True)
                self._take_step(e_loss, self._opt_e, self._gs_e, self._lrs_e, retain_graph=True)
                self._take_step(e_loss, self._opt_r, self._gs_r, self._lrs_r, retain_graph=True)
                if dist.is_main_process():
                    dataloader.set_description(f'{descr} Emb: recon_loss: {e_loss.item():.4f}')
            dist.barrier()
            if dist.is_main_process():
                self._save_checkpoint(0, 'final', 0, 0)

        if dist.is_main_process():
            self._save_checkpoint(0, 'final', 0, 0)
        self._pretrained = True, False
        _LOGGER.debug('Finished pretraining embedding.')

    def _recover(self, data: Tensor) -> (Tensor, Tensor):
        h, _ = self._embedder(data)
        h = h.view(*data.shape[:2], self._hidden_dim)
        x_tilde, _ = self._recovery(h)
        x_tilde = x_tilde.view(*data.shape[:2], -1)
        return h, x_tilde

    def _run_supervised(self, dataloader: DataLoader, epochs: int = 10):
        if self._pretrained[1]:
            return
        self._embedder.train()
        self._supervisor.train()
        for i in range(epochs):
            descr = f'Epoch[{i}] {self._descr} Sup'
            if dist.is_main_process():
                dataloader = tqdm(dataloader)
                dataloader.set_description(descr)
            for step, (_, known_batch, unknown_batch) in enumerate(dataloader):
                data = torch.cat([known_batch, unknown_batch], dim=-1)
                h, _ = self._embedder(data)
                h = h.view(*data.shape[:2], self._hidden_dim)
                h_hat, _ = self._supervisor(h)
                h_hat = h_hat.view(*data.shape[:2], -1)

                g_loss_s = F.mse_loss(h[:, 1:, :], h_hat[:, :-1, :])
                self._take_step(g_loss_s, self._opt_e, self._gs_e, self._lrs_e, retain_graph=True)
                self._take_step(g_loss_s, self._opt_s, self._gs_s, self._lrs_s, retain_graph=True)
                if dist.is_main_process():
                    dataloader.set_description(f'{descr} Sup: sup_loss: {g_loss_s.item():.4f}')
            dist.barrier()
            if dist.is_main_process():
                self._save_checkpoint(0, 'final', 0, 0)

        if dist.is_main_process():
            self._save_checkpoint(0, 'final', 0, 0)
        self._pretrained = True
        _LOGGER.debug('Finished pretraining supervised')

    def _collate_fn(self, batch: List[Tuple[Tensor, ...]]) -> Tuple[Tensor, ...]:
        known, unknown = super()._collate_fn(batch)
        noise = self._get_noise(len(batch), known)
        return noise, known, unknown

    def _collate_fn_infer(self, batch: List[Tuple[Tensor, ...]]) -> Tuple[Tensor, ...]:
        known, = super()._collate_fn_infer(batch)
        noise = self._get_noise(len(batch), known)
        return noise, known

    def _get_noise(self, batch_size: int, known: Tensor) -> Tensor:
        z_dim = self._unknown_dim + self._known_dim + 1
        noise = []
        for i in range(batch_size):
            temp_z = np.random.uniform(0., 10, (known.shape[-2], z_dim))
            noise.append(torch.tensor(temp_z))
            # seq_len = known.shape[-2]
            # temp = torch.zeros(known.shape[1], z_dim)
            # temp_z = np.random.uniform(0., 10, (seq_len, z_dim))
            # temp[:seq_len] = torch.tensor(temp_z)
            # print('shapes:', temp.shape, known.shape, flush=True)
            # temp[:, :self._known_dim] = known
            # noise.append(temp)
        return torch.stack(noise)

    def _prepare_epoch(self, dataloader: DataLoader, base_step: int, global_step: int, save_freq: int):
        if base_step < global_step:
            return
        if base_step == global_step:
            self._reload_checkpoint(global_step // save_freq, 'step')
        for prepare in range(2):
            batch = next(iter(dataloader))
            batch = tuple(to_device(b, self._device) for b in batch)
            noise, known, unknown = batch
            batch_size = noise.shape[0]

            data = torch.cat([known, unknown], dim=-1)
            g_loss_s, g_loss_u, g_loss_v1, g_loss_v2, g_loss_v = self._joint_step(batch_size, noise, data, unknown)
            self._take_step(g_loss_s + g_loss_u + g_loss_v, self._opt_g, self._gs_g, self._lrs_g,
                            retain_graph=True)
            self._take_step(g_loss_s + g_loss_u + g_loss_v, self._opt_s, self._gs_s, self._lrs_s,
                            retain_graph=True)
            self._take_step(g_loss_s + g_loss_u + g_loss_v, self._opt_d, self._gs_d, self._lrs_d,
                            retain_graph=True)

            h, x_tilde = self._recover(data)
            e_loss = self._calculate_recon_loss(data[:, :, known.shape[-1]:], x_tilde, True) / 10
            h_sup, _ = self._supervisor(h)
            h_sup = h_sup.view(batch_size, -1, self._hidden_dim)
            g_loss_s = F.mse_loss(h[:, 1:, :], h_sup[:, :-1, :])
            self._take_step(e_loss / 10 + g_loss_s, self._opt_e, self._gs_e, self._lrs_e, retain_graph=True)
            self._take_step(e_loss / 10 + g_loss_s, self._opt_r, self._gs_r, self._lrs_r, retain_graph=True)
            self._take_step(e_loss / 10 + g_loss_s, self._opt_s, self._gs_s, self._lrs_s, retain_graph=True)

    def _joint_step(self, batch_size: int, noise: Tensor, data: Tensor, unknown: Tensor) -> (
            Tensor, Tensor, Tensor, Tensor, Tensor):
        e_hat, _ = self._generator(noise)
        e_hat = e_hat.view(batch_size, -1, self._hidden_dim)
        h_hat, _ = self._supervisor(e_hat)
        h_hat = h_hat.view(batch_size, -1, self._hidden_dim)
        y_fake = self._discriminator(h_hat)
        y_fake = y_fake.view(batch_size, -1, 1)
        x_hat, _ = self._recovery(h_hat)
        x_hat = x_hat.view(batch_size, -1, self._unknown_dim + 1)
        h, _ = self._embedder(data)
        h = h.view(batch_size, -1, self._hidden_dim)
        h_sup, _ = self._supervisor(h)
        h_sup = h_sup.view(batch_size, -1, self._hidden_dim)
        g_loss_s = F.mse_loss(h[:, 1:, :], h_sup[:, :-1, :])
        g_loss_u = F.binary_cross_entropy_with_logits(y_fake, torch.ones_like(y_fake))
        g_loss_v1 = (x_hat.std([0], unbiased=False).abs() + 1e-6 - (unknown.std([0]) + 1e-6)).mean()
        g_loss_v2 = (x_hat.mean([0]) - unknown.mean([0])).abs().mean()
        g_loss_v = g_loss_v1 + g_loss_v2
        return g_loss_s, g_loss_u, g_loss_v1, g_loss_v2, g_loss_v

    def run_step(self, batch: Tuple[Tensor, ...]) -> Tuple[Dict[str, float], Optional[Tensor]]:
        noise, known, unknown = batch
        data = torch.cat([known, unknown], dim=-1)
        batch_size = noise.shape[0]
        h, _ = self._embedder(data)
        h = h.view(batch_size, -1, self._hidden_dim)
        y_real = self._discriminator(h).view(batch_size, -1, 1)
        e_hat, _ = self._generator(noise)
        e_hat = e_hat.view(batch_size, -1, self._hidden_dim)
        y_fake_e = self._discriminator(e_hat).view(batch_size, -1, 1)
        h_hat, _ = self._supervisor(e_hat)
        h_hat = h_hat.view(batch_size, -1, self._hidden_dim)
        y_fake = self._discriminator(h_hat).view(batch_size, -1, 1)
        x_hat, _ = self._recover(h_hat)
        x_hat = x_hat.view(batch_size, -1, self._unknown_dim + 1)

        self._generator.zero_grad()
        self._supervisor.zero_grad()
        self._discriminator.zero_grad()
        self._recovery.zero_grad()
        self._embedder.zero_grad()

        dlr = F.binary_cross_entropy_with_logits(y_real, torch.ones_like(y_real))
        dlf = F.binary_cross_entropy_with_logits(y_fake, torch.zeros_like(y_fake))
        dlf_e = F.binary_cross_entropy_with_logits(y_fake_e, torch.zeros_like(y_fake_e))
        d_loss = dlr + dlf + self._gamma * dlf_e
        if d_loss > 0.15:
            self._take_step(d_loss, self._opt_d, self._gs_d, self._lrs_d, retain_graph=True)

        h, x_tilde = self._recover(data)
        g_loss_s, g_loss_u, g_loss_v1, g_loss_v2, g_loss_v = self._joint_step(batch_size, noise, data, unknown)
        e_loss = self._calculate_recon_loss(unknown, x_tilde, True)
        mean_l, corr_l = self._meta_loss(known, unknown, x_hat)
        recon_loss = self._calculate_recon_loss(unknown, x_hat, True)
        self._take_step(g_loss_s, None, None, None, retain_graph=True, do_zero_grad=False, do_step=False)
        self._take_step(g_loss_u, None, None, None, retain_graph=True, do_zero_grad=False, do_step=False)
        self._take_step(g_loss_v, None, None, None, retain_graph=True, do_zero_grad=False, do_step=False)
        self._take_step(e_loss + 0.1 * g_loss_s, None, None, None, retain_graph=True, do_zero_grad=False, do_step=False)
        self._take_step(recon_loss, None, None, None, retain_graph=True, do_zero_grad=False, do_step=False)
        self._take_step(mean_l + corr_l, self._opt_g, self._gs_g, self._lrs_g, do_step=True,
                        do_zero_grad=False, do_backward=True)
        self._take_step(None, self._opt_s, self._gs_s, self._lrs_s, do_step=True, do_zero_grad=False, do_backward=False)
        self._take_step(None, self._opt_e, self._gs_e, self._lrs_e, do_step=True, do_zero_grad=False, do_backward=False)
        self._take_step(None, self._opt_r, self._gs_r, self._lrs_r, do_step=True, do_zero_grad=False, do_backward=False)

        out_dict = {
            'd': d_loss,
            'gs': g_loss_s,
            'gu': g_loss_u,
            'gv': g_loss_v,
            'e': e_loss,
            'm': mean_l,
            'c': corr_l
        }
        out_dict = {k: v.detach().cpu().item() for k, v in out_dict.items()}
        out_dict['elr'] = self._opt_e.param_groups[0]['lr']
        out_dict['rlr'] = self._opt_r.param_groups[0]['lr']
        out_dict['glr'] = self._opt_g.param_groups[0]['lr']
        out_dict['slr'] = self._opt_s.param_groups[0]['lr']
        out_dict['dlr'] = self._opt_d.param_groups[0]['lr']
        return out_dict, x_hat[:, :, :-1]

    def _meta_loss(self, known: Tensor, real: Tensor, fake: Tensor) -> (Tensor, Tensor):
        known, _ = self._expand_series(known, real[:, :, -1])
        real, _ = self._expand_series(real)
        fake, _ = self._expand_series(fake[:, :, :-1], real[:, :, -1])
        return self._meta_loss(known, real, fake)

    @staticmethod
    def _expand_series(x: Tensor, indicator: Optional[Tensor] = None) -> (Tensor, List[int]):
        if indicator is None:
            indicator = x[:, :, -1]
            x = x[:, :, :-1]
        out = []
        lengths = []
        for seq, ind in (x, indicator):
            maintained = (ind > 0.5).tolist()
            width = 5
            length = len(seq)
            while width > 0:
                find = maintained.index([False] * width)
                if find > 0:
                    seq = seq[:find]
                    length = find
                    break
                else:
                    width -= 1
            out.append(seq)
            lengths.append(length)
        return torch.cat(out), lengths

    def inference(self, known: Tensor, batch_size: int = 500) -> TimeGANOutput:
        dataloader = self._make_infer_dataloader(known, batch_size, False)
        autocast = torch.cuda.is_available() and self._autocast
        if dist.is_main_process():
            dataloader = tqdm(dataloader)
            dataloader.set_description(f'Inference on {self._descr}')

        self._embedder.eval()
        self._recovery.eval()
        self._generator.eval()
        self._supervisor.eval()
        self._discriminator.eval()
        fakes, y_fakes, lengths = [], [], []
        for step, (noise, known) in enumerate(dataloader):
            with torch.cuda.amp.autocast(enabled=autocast):
                e_hat, _ = self._generator(noise)
                e_hat = e_hat.view(batch_size, -1, self._hidden_dim)
                h_hat, _ = self._supervisor(e_hat)
                h_hat = h_hat.view(batch_size, -1, self._hidden_dim)
                y_fake = self._discriminator(h_hat).view(batch_size, -1, 1)
                x_hat, _ = self._recover(h_hat)
                x_hat = x_hat.view(batch_size, -1, self._unknown_dim + 1)
                fake = torch.cat([known, x_hat], dim=-1)
                y_fake, lengths = self._expand_series(y_fake, fake[:, :, -1])
                fake, _ = self._expand_series(fake)
                fakes.extend(fake)
                y_fakes.extend(y_fake)
                lengths.append(lengths)

        self._embedder.train()
        self._recovery.train()
        self._generator.train()
        self._supervisor.train()
        self._discriminator.train()

        return TimeGANOutput(
            fake=torch.cat(fakes),
            discr_out=torch.cat(y_fakes),
            lengths=lengths
        )
