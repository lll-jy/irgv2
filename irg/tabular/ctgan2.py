import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from ctgan.data_sampler import DataSampler as CTGANDataSampler
from ctgan.data_transformer import SpanInfo
from ctgan.synthesizers.ctgan import Generator, Discriminator
from sklearn.neighbors import KNeighborsClassifier
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from packaging import version

from .base import TabularTrainer
from ..utils import InferenceOutput
from ..utils.dist import is_main_process

_LOGGER = logging.getLogger()


class DataSampler(CTGANDataSampler):
    def __init__(self, data: Tensor, context: Tensor, info: List[List[SpanInfo]]):
        super().__init__(data, info, True)

        self._has_context = context.shape[1] > 0
        if self._has_context:
            self._knn_context = []
            st = 0
            current_id = 0
            current_cond_st = 0
            for column_info in info:
                if self._is_discrete_column(column_info):
                    span_info = column_info[0]
                    ed = st + span_info.dim
                    y = data[:, st:ed].argmax(axis=1)
                    knn = KNeighborsClassifier()
                    knn.fit(context, y)
                    self._knn_context.append(knn)
                    current_cond_st += span_info.dim
                    current_id += 1
                    st = ed
                else:
                    st += sum([span_info.dim for span_info in column_info])
        else:
            self._knn_context = None

        delattr(self, '_data')

    @staticmethod
    def _is_discrete_column(column_info: List[SpanInfo]):
        return (len(column_info) == 1
                and column_info[0].activation_fn == 'softmax')

    def sample_condvec(self, unknown: Tensor) -> (Tensor, Tensor, Tensor, Tensor):
        if self._n_discrete_columns == 0:
            return [torch.zeros(unknown.shape[0], 0) for _ in range(4)]
        batch = unknown.shape[0]
        discrete_column_id = torch.from_numpy(np.random.choice(
            np.arange(self._n_discrete_columns), batch))

        mask = np.zeros(batch, self._n_discrete_columns, dtype=torch.float32)
        mask[np.arange(batch), discrete_column_id] = 1
        cond = torch.zeros(batch, self._n_categories, dtype=torch.float32)
        st = self._discrete_column_cond_st[discrete_column_id]
        width = self._discrete_column_n_category[discrete_column_id]
        category_id_in_col = []
        for i, row_st, row_width, unknown_row in zip(range(batch), st, width, unknown):
            cond[i, st:st+width] = unknown_row[st:st+width]
            category_id_in_col.append(unknown_row[st:st+width].argmax(dim=-1))
        category_id_in_col = torch.tensor(category_id_in_col)
        return cond, mask, discrete_column_id, category_id_in_col

    def sample_original_condvec(self, known: Tensor) -> Tensor:
        if self._n_discrete_columns == 0:
            return torch.zeros(*known.shape)
        cond = torch.zeros(known.shape[0], self._n_categories, dtype=torch.float32)
        discrete_column_id = torch.from_numpy(np.random.choice(
            np.arange(self._n_discrete_columns), known.shape[0]))
        for i, row in enumerate(known):
            discrete_col = discrete_column_id[i]
            choice = self._choice_index(discrete_col, row)
            st = self._discrete_column_cond_st[discrete_col]
            width = self._discrete_column_n_category[discrete_col]
            empty = torch.zeros(width)
            empty[choice] = 1
            cond[i, st:st+width] = empty
        return cond

    def _choice_index(self, discrete_column_id: int, known_row: Tensor) -> np.ndarray:
        if not self._has_context:
            return super()._random_choice_prob_index(discrete_column_id)
        knn: KNeighborsClassifier = self._knn_context[discrete_column_id]
        known = torch.stack([known_row])
        pred = knn.predict(known)
        return pred


class CTGANOutput(InferenceOutput):
    def __init__(self, fake: Tensor, discr_out: Optional[Tensor] = None):
        super().__init__(fake)
        self.fake = fake
        """Fake data generated."""
        self.discr_out = discr_out
        """Discriminator output."""


class CTGANTrainer(TabularTrainer):
    def __init__(self, embedding_dim: int = 128,
                 generator_dim: Tuple[int, ...] = (256, 256), discriminator_dim: Tuple[int, ...] = (256, 256),
                 pac: int = 10, discriminator_step: int = 1, **kwargs):
        super().__init__(**{
            n: v for n, v in kwargs.items() if
            n in {'distributed', 'autocast', 'log_dir', 'ckpt_dir', 'descr',
                  'cat_dims', 'known_dim', 'unknown_dim'}
        })

        self._info_list, self._cond_dim = [], 0
        self._learn_condvec_info()
        self._encoded_dim = 0 if self._known_dim == 0 else self._lae.encoded_dim

        self._sampler: Optional[DataSampler] = None
        self._generator = Generator(
            embedding_dim=embedding_dim + self._encoded_dim + self._cond_dim,
            generator_dim=generator_dim,
            data_dim=self._unknown_dim
        ).to(self._device)
        self._discriminator = Discriminator(
            input_dim=self._encoded_dim + self._cond_dim + self._unknown_dim,
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

    def _load_content_from(self, loaded: Dict[str, Any]):
        super()._load_content_from(loaded)
        self._generator, self._optimizer_g, self._lr_schd_g, self._grad_scaler_g = self._load_state_dict(
            self._generator, self._optimizer_g, self._lr_schd_g, self._grad_scaler_g, loaded['generator']
        )
        self._discriminator, self._optimizer_d, self._lr_schd_d, self._grad_scaler_d = self._load_state_dict(
            self._discriminator, self._optimizer_d, self._lr_schd_d, self._grad_scaler_d, loaded['discriminator']
        )
        self._sampler = loaded['sampler']

    def _construct_content_to_save(self) -> Dict[str, Any]:
        return {
                   'generator': self._full_state_dict(
                       self._generator, self._optimizer_g, self._lr_schd_g, self._grad_scaler_g
                   ),
                   'discriminator': self._full_state_dict(
                       self._discriminator, self._optimizer_d, self._lr_schd_d, self._grad_scaler_d
                   ),
                   'sampler': self._sampler
               } | super()._construct_content_to_save()

    @classmethod
    def _reconstruct(cls, distributed: bool, autocast: bool, log_dir: str, ckpt_dir: str, descr: str,
                     known_dim: int, unknown_dim: int, cat_dims: List[Tuple[int, int]], lae_trained: bool,
                     lae: nn.Module, optimizer_lae: Optimizer, lr_schd_lae: LRScheduler,
                     grad_scaler_lae: Optional[GradScaler], info_list: List[List[SpanInfo]], cond_dim: int,
                     encoded_dim: int, sampler: CTGANDataSampler, generator: nn.Module, optimizer_g: Optimizer,
                     lr_schd_g: LRScheduler, grad_scaler_g: Optional[GradScaler], discriminator: nn.Module,
                     optimizer_d: Optimizer, lr_schd_d: LRScheduler, grad_scaler_d: Optional[GradScaler],
                     embedding_dim: int, pac: int, discriminator_step: int) -> "CTGANTrainer":
        base = TabularTrainer._reconstruct(
            distributed, autocast, log_dir, ckpt_dir, descr,
            known_dim, unknown_dim, cat_dims, lae_trained,
            lae, optimizer_lae, lr_schd_lae, grad_scaler_lae
        )
        base.__class__ = CTGANTrainer
        base._info_list, base._cond_dim, base._encoded_dim = info_list, cond_dim, encoded_dim
        base._sampler = sampler
        base._generator, base._optimizer_g, base._lr_schd_g, base._grad_scaler_g = (
            generator, optimizer_g, lr_schd_g, grad_scaler_g)
        base._discriminator, base._optimizer_d, base._lr_schd_d, base._grad_scaler_d = (
            discriminator, optimizer_d, lr_schd_d, grad_scaler_d)

        base._embedding_dim, base._pac, base._discriminator_step = embedding_dim, pac, discriminator_step
        return base

    def __reduce__(self):
        _, var = super().__reduce__()
        return self._reconstruct, var + (
            self._info_list, self._cond_dim, self._encoded_dim, self._sampler,
            self._generator, self._optimizer_g, self._lr_schd_g, self._grad_scaler_g,
            self._discriminator, self._optimizer_d, self._lr_schd_d, self._grad_scaler_d,
            self._embedding_dim, self._pac, self._discriminator_step
        )

    def _learn_condvec_info(self):
        is_for_num, ptr, cat_ptr = False, 0, 0
        info_list = []
        cond_dim = 0
        while ptr < self._unknown_dim:
            if cat_ptr < len(self._cat_dims) and ptr == self._cat_dims[cat_ptr][0]:
                l, r = self._cat_dims[cat_ptr]
                cat_ptr += 1
                if r - l > 1 and not is_for_num:
                    info_list.append([SpanInfo(r-l, 'softmax')])
                    cond_dim += r - l
                elif is_for_num:
                    info_list.append([SpanInfo(1, 'tanh'), SpanInfo(r-l, 'softmax')])
                else:
                    info_list.append([SpanInfo(1, 'sigmoid')])
                is_for_num = False
                ptr = r
            else:
                ptr += 1
                is_for_num = True
        self._info_list = info_list
        self._cond_dim = cond_dim

    def _construct_sampler(self, known: Tensor, unknown: Tensor):
        self._sampler = DataSampler(
            data=unknown,
            context=known,
            info=self._info_list
        )

    def _collate_fn(self, batch: List[Tuple[Tensor, ...]]) -> Tuple[Tensor, ...]:
        batch_size = len(batch)
        known, unknown = super()._collate_fn(batch)
        mean = torch.zeros(batch_size, self._embedding_dim)
        std = mean + 1

        disc_in_fakez, disc_in_c, disc_in_perm = [], [], []
        for ds in range(self._discriminator_step):
            fakez = torch.normal(mean=mean, std=std)
            c1, m1, col, opt = self._sampler.sample_condvec(unknown)
            perm = np.arange(batch_size)
            np.random.shuffle(perm)
            perm = torch.from_numpy(perm)
            disc_in_fakez.append(fakez)
            disc_in_c.append(c1)
            disc_in_perm.append(perm)
        disc_in_fakez = torch.stack(disc_in_fakez)
        disc_in_c = torch.stack(disc_in_c)
        disc_in_perm = torch.stack(disc_in_perm)

        gen_in_fakez = torch.normal(mean=mean, std=std)
        c1, m1, col, opt = self._sampler.sample_condvec(unknown)

        return known, unknown, disc_in_fakez, disc_in_c, disc_in_perm, gen_in_fakez, c1, m1

    def _collate_fn_infer(self, batch: List[Tuple[Tensor, ...]]) -> Tuple[Tensor, ...]:
        batch_size = len(batch)
        known, = super()._collate_fn_infer(batch)
        mean = torch.zeros(batch_size, self._embedding_dim)
        std = mean + 1
        fakez = torch.normal(mean=mean, std=std)
        condvec = self._sampler.sample_original_condvec(known)
        return known, fakez, condvec

    def train(self, known: Tensor, unknown: Tensor, epochs: int = 10, batch_size: int = 100, shuffle: bool = True,
              save_freq: int = 100, resume: bool = True, lae_epochs: int = 10):
        self._construct_sampler(known, unknown)
        _LOGGER.debug(f'Constructed data sampler for {self._descr}.')
        super().train(known, unknown, epochs, batch_size, shuffle, save_freq, resume, lae_epochs)

    def run_step(self, batch: Tuple[Tensor, ...]) -> Tuple[Dict[str, float], Optional[Tensor]]:
        enable_autocast = torch.cuda.is_available() and self._autocast
        known, unknown, disc_in_fakez, disc_in_c, disc_in_perm, gen_in_fakez, gen_cond, gen_mask = batch
        known = self._make_context(known)

        for fakez, c1, perm in zip(disc_in_fakez, disc_in_c, disc_in_perm):
            with torch.cuda.amp.autocast(enabled=enable_autocast):
                real_cat = torch.cat([known, c1, unknown], dim=1)[perm]
                fake_cat = self._construct_fake(known, c1, fakez)
                real_cat = self._make_full_pac(real_cat)
                fake_cat = self._make_full_pac(fake_cat)
                y_fake = self._discriminator(fake_cat)
                y_real = self._discriminator(real_cat)
                pen = self._discriminator.calc_gradient_penalty(
                    real_cat, fake_cat, self._device, self._pac
                )
                mse_d = self._mse_loss(fake_cat[perm], real_cat)
                loss_d = -(torch.mean(y_real) - torch.mean(y_fake))
                loss = mse_d + loss_d
            self._optimizer_d.zero_grad()
            if self._grad_scaler_d is None:
                pen.backward(retain_graph=True)
            else:
                self._grad_scaler_d.scale(pen).backward(retain_graph=True)
            self._take_step(loss, self._optimizer_d, self._grad_scaler_d, self._lr_schd_d, do_zero_grad=False)

        with torch.cuda.amp.autocast(enabled=enable_autocast):
            real_cat = torch.cat([known, gen_cond, unknown], dim=1)
            fake_cat = self._construct_fake(known, gen_cond, gen_in_fakez)
            y_fake = self._discriminator(fake_cat)
            loss_g = -torch.mean(y_fake)
            cross_entropy = self._cond_loss(fake_cat[:, -self._unknown_dim:], gen_cond, gen_mask)
            mse_g = self._mse_loss(fake_cat, real_cat)
            loss = loss_g + cross_entropy + mse_g
        self._take_step(loss, self._optimizer_g, self._grad_scaler_g, self._lr_schd_g)
        return {
            'g': loss_g.detach().cpu().item(),
            'd': loss_d.detach().cpu().item(),
            'ce': cross_entropy.detach().cpu().item(),
            'dg': mse_g.detach().cpu().item(),
            'dd': mse_d.detach().cpu().item(),
            'pen': pen.detach().cpu().item(),
        }, fake_cat

    def _mse_loss(self, fake_cat: Tensor, real_cat: Tensor) -> Tensor:
        if self._known_dim > 0:
            return F.mse_loss(
                input=fake_cat[:, -self._unknown_dim:],
                target=real_cat[:, -self._unknown_dim:],
                reduction='mean'
            )
        else:
            return torch.tensor(0).to(self._device)

    def _cond_loss(self, data: Tensor, c: Tensor, m: Tensor):
        if c.shape[1] == 0:
            return torch.tensor(0).to(self._device)
        loss = []
        st = 0
        st_c = 0
        for column_info in self._info_list:
            for span_info in column_info:
                if len(column_info) != 1 or span_info.activation_fn != 'softmax':
                    # not discrete column
                    st += span_info.dim
                else:
                    ed = st + span_info.dim
                    ed_c = st_c + span_info.dim
                    tmp = F.cross_entropy(
                        data[:, st:ed],
                        torch.argmax(c[:, st_c:ed_c], dim=1),
                        reduction='none'
                    )
                    loss.append(tmp)
                    st = ed
                    st_c = ed_c

        loss = torch.stack(loss, dim=1)  # noqa: PD013

        return (loss * m).sum() / data.size()[0]

    def _make_full_pac(self, x: Tensor) -> Tensor:
        n, m = x.shape
        size = math.ceil(n / self._pac) * self._pac
        y = torch.zeros(size, m).to(x.device)
        y[:n, :] = x
        return y

    def _construct_fake(self, known: Tensor, cond: Tensor, noise: Tensor) -> Tensor:
        fake = self._generator(torch.cat([noise, cond, known], dim=1))
        fakeact = self._apply_activate(fake)
        fake_cat = torch.cat([known, cond, fakeact], dim=1)
        return fake_cat

    def _apply_activate(self, data: Tensor):
        """Apply proper activation function to the output of the generator."""
        data_t = []
        st = 0
        for column_info in self._info_list:
            for span_info in column_info:
                ed = st + span_info.dim
                if span_info.activation_fn == 'tanh':
                    data_t.append(torch.tanh(data[:, st:ed]))
                elif span_info.activation_fn == 'softmax':
                    transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2)
                    data_t.append(transformed)
                elif span_info.activation_fn == 'sigmoid':
                    data_t.append(torch.sigmoid(data[:, st:ed]))
                else:
                    raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')
                st = ed

        return torch.cat(data_t, dim=1)

    @staticmethod
    def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
        if version.parse(torch.__version__) < version.parse('1.2.0'):
            for i in range(10):
                transformed = F.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=-1)
                if not torch.isnan(transformed).any():
                    return transformed
            raise ValueError('gumbel_softmax returning NaN.')

        return F.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=-1)

    @torch.no_grad()
    def inference(self, known: Tensor, batch_size: int) -> InferenceOutput:
        dataloader = self._make_infer_dataloader(known, batch_size, False)
        autocast = torch.cuda.is_available() and self._autocast
        if is_main_process():
            dataloader = tqdm(dataloader)
            dataloader.set_description(f'Inference on {self._descr}')

        fakes, y_fakes = [], []
        self._generator.eval()
        self._discriminator.eval()
        for step, batch in enumerate(dataloader):
            known_batch, fakez, condvec = [x.to(self._device) for x in batch]
            with torch.cuda.amp.autocast(enabled=autocast):
                known = self._make_context(known_batch)
                fake_cat = self._construct_fake(known, condvec, fakez)
                y_fake = self._discriminator(fake_cat)
                y_fake = y_fake.repeat(self._pac, 1).permute(1, 0).flatten()[:fake_cat.shape[0]]
                fakes.append(fake_cat)
                y_fakes.append(y_fake)
        self._generator.train()
        self._discriminator.train()

        return CTGANOutput(torch.cat(fakes), torch.cat(y_fakes))
