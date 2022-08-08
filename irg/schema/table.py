import json
import os
from typing import Optional, Iterable, Dict, Tuple, Set, List

import torch
from torch import Tensor
import pandas as pd
from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator

from .attribute import learn_meta, create as create_attribute, BaseAttribute, SerialIDAttribute
from ..utils.misc import Data2D, convert_data_as, inverse_convert_data
from ..utils.errors import NoPartiallyKnownError, NotFittedError

TwoLevelName = Tuple[str, str]


class Table:
    def __init__(self, name: str, need_fit: bool = True, id_cols: Iterable[str] = None,
                 attributes: Optional[Dict[str, dict]] = None, data: Optional[pd.DataFrame] = None,
                 determinants: Optional[List[List[str]]] = None, formulas: Optional[Dict[str, str]] = None, **kwargs):
        self._name, self._need_fit, self._fitted = name, need_fit, False
        self._determinants = [] if determinants is None else determinants
        self._formulas = {} if formulas is None else formulas
        id_cols = set() if id_cols is None else set(id_cols)
        if attributes is None:
            if data is None:
                raise ValueError('Data and attributes cannot both be `None` to create a table.')
            attributes = {
                attr_name: learn_meta(data[attr_name], attr_name in id_cols, attr_name) | {'name': attr_name}
                for attr_name in data.columns
            }
        self._attr_meta, self._data, self._id_cols = attributes, data, id_cols
        self._attributes: Dict[str, BaseAttribute] = {
            attr_name: create_attribute(meta, data[attr_name] if need_fit and data is not None else None)
            for attr_name, meta in self._attr_meta.items()
        }

        det_child_cols = {col for det in self._determinants for col in det[1:]}
        self._core_cols = [
            col for col in self._attributes
            if col not in det_child_cols and col not in self._formulas
        ]

        self._normalized_by_attr = {}
        self._describers = []
        if need_fit and data is not None:
            self.fit(data, **kwargs)
        if self._fitted:
            self._normalized_by_attr = {
                attr_name: data[[attr_name]] if attr_name in self._id_cols else attr.get_original_transformed()
                for attr_name, attr in self._attributes.items()
            }

        self._known_cols, self._unknown_cols, self._augment_fitted = [], [*self._attributes.keys()], False
        self._augmented: Optional[pd.DataFrame] = None
        self._degree: Optional[pd.DataFrame] = None
        self._augmented_meta: Dict[TwoLevelName, dict] = {}
        self._degree_meta: Dict[TwoLevelName, dict] = {}
        self._augmented_attributes: Dict[TwoLevelName, BaseAttribute] = {}
        self._degree_attributes: Dict[TwoLevelName, BaseAttribute] = {}
        self._augmented_normalized_by_attr: Dict[TwoLevelName, pd.DataFrame] = {}
        self._degree_normalized_by_attr: Dict[TwoLevelName, pd.DataFrame] = {}
        self._augmented_ids: Set[TwoLevelName] = set()
        self._degree_ids: Set[TwoLevelName] = set()

    @property
    def name(self) -> str:
        return self._name

    def fit(self, data: pd.DataFrame, force_redo: bool = False, **kwargs):
        if (self._fitted and not force_redo) or not self._need_fit:
            return
        self._data = data[[*self._attributes.keys()]]
        for name, attr in self._attributes.items():
            attr.fit(data[name], force_redo=force_redo)
            self._normalized_by_attr[name] = attr.get_original_transformed()

        for det in self._determinants:
            for col in det:
                if self._attributes[col].atype != 'categorical':
                    raise TypeError('Determinant should have all columns categorical.')
            leader, children = det[0], det[1:]
            det_describers = self._fit_determinant(leader, children, **kwargs)
            self._describers.append(det_describers)
        self._fitted = True

    def _fit_determinant(self, leader: str, children: List[str], **kwargs) -> Dict[str, Dict]:
        describers = dict()
        for grp_name, data in self._data.groupby(by=[leader], sort=False, dropna=False):
            describer = DataDescriber(**kwargs)
            if len(data) <= 1:
                data = pd.concat([data, data]).reset_index(drop=True)
            if pd.isnull(grp_name):
                grp_name = self._attributes[leader].fill_nan_val
            data = data.copy()
            data[leader] = grp_name
            for col in children:
                data[col] = data[col].fillna(self._attributes[col].fill_nan_val)
            data[':dummy'] = 'dummy'
            tempfile_name = f'{self._name}__{leader}__{grp_name}'
            data.to_csv(f'{tempfile_name}.csv', index=False)
            describer.describe_dataset_in_correlated_attribute_mode(
                f'{tempfile_name}.csv', k=len(children), epsilon=0,
                attribute_to_datatype={col: 'String' for col in children} | {leader: 'String'},
                attribute_to_is_categorical={col: True for col in children} | {leader: True},
                attribute_to_is_candidate_key={col: False for col in children} | {leader: False}
            )
            describer.save_dataset_description_to_file(f'{tempfile_name}.json')
            with open(f'{tempfile_name}.json', 'r') as f:
                describer_info = json.load(f)
                f.close()
            os.remove(f'{tempfile_name}.csv')
            os.remove(f'{tempfile_name}.json')
            describers[grp_name] = describer_info
        return describers

    def data(self, variant: str = 'original', normalize: bool = False,
             with_id: str = 'this', core_only: bool = False, return_as: str = 'pandas') -> Data2D:
        if with_id not in {'this', 'none', 'inherit'}:
            raise NotImplementedError(f'With id policy "{with_id}" is not recognized.')
        if self._data is None:
            raise NotFittedError('Table', 'getting its data')
        if variant == 'original':
            exclude_cols = self._id_cols if with_id == 'none' else set()
            exclude_cols |= {col for col in self._attributes if col not in self._core_cols} if core_only else set()
            if not normalize:
                data = self._data[[col for col in self._data.columns if col not in exclude_cols]]
            else:
                data = pd.concat({n: v for n, v in self._normalized_by_attr.items() if n not in exclude_cols}, axis=1)
        elif variant == 'augmented':
            data = self._get_aug_or_deg_data(self._augmented, self._augmented_normalized_by_attr,
                                             self._augmented_ids, normalize, with_id, core_only)
        elif variant == 'degree':
            data = self._get_aug_or_deg_data(self._degree, self._degree_normalized_by_attr, self._degree_ids,
                                             normalize, with_id, core_only)
        else:
            raise NotImplementedError(f'Getting data variant "{variant}" is not recognized.')

        return convert_data_as(data, return_as=return_as, copy=True)

    def _get_aug_or_deg_data(self, data: pd.DataFrame, normalized_by_attr: Dict[TwoLevelName, pd.DataFrame],
                             id_cols: Set[TwoLevelName], normalize: bool = False, with_id: str = 'this',
                             core_only: bool = False) -> pd.DataFrame:
        if self.is_independent:
            raise NoPartiallyKnownError(self._name)
        if with_id == 'inherit':
            exclude_cols = set()
        elif with_id == 'this':
            exclude_cols = {(table, attr) for table, attr in id_cols if table != self._name}
        else:
            assert with_id == 'none'
            exclude_cols = id_cols
        if core_only:
            exclude_cols |= {
                (table, attr) for table, attr in normalized_by_attr
                if table == self._name and attr not in self._core_cols
            }
        if not normalize:
            data = data[[col for col in data.columns if col not in exclude_cols]]
        else:
            data = pd.concat({n: v for n, v in normalized_by_attr.items() if n not in exclude_cols}, axis=1)
        return data

    @property
    def is_independent(self):
        return not self._augment_fitted

    @property
    def ptg_data(self) -> Tuple[Tensor, Tensor]:
        if not self.is_independent:
            unknown_cols = [
                (table, attr) for table, attr in self._augmented_attributes
                if table == self._name and attr not in self._known_cols
            ]
            aug_data = self.data(variant='augmented', normalize=True, with_id='none', core_only=True)
            unknown_set = set(unknown_cols)
            known_cols = [col for col in aug_data.columns.droplevel(2) if col not in unknown_set]
            known_data, unknown_data = aug_data[known_cols], aug_data[unknown_cols]
            return convert_data_as(known_data, 'torch'), convert_data_as(unknown_data, 'torch')
        else:
            norm_data = self.data(variant='original', normalize=True, with_id='none', core_only=True)
            return torch.zeros(len(norm_data), 0), convert_data_as(norm_data, 'torch')

    @property
    def deg_data(self) -> Tuple[Tensor, Tensor]:
        if self.is_independent:
            raise NoPartiallyKnownError(self._name)
        unknown_cols = [
            (table, attr) for table, attr in self._degree_attributes
            if table == self._name and attr not in self._known_cols
        ]
        deg_data = self.data(variant='degree', normalize=True, with_id='none', core_only=True)
        unknown_set = set(unknown_cols)
        known_cols = [col for col in deg_data.columns.droplevel(2) if col not in unknown_set]
        known_data, unknown_data = deg_data[unknown_cols], deg_data[known_cols]
        return convert_data_as(known_data, 'torch'), convert_data_as(unknown_data, 'torch')


class SyntheticTable(Table):
    @classmethod
    def from_real(cls, table: Table) -> "SyntheticTable":
        synthetic = SyntheticTable(name=table._name, need_fit=False,
                                   id_cols={*table._id_cols}, attributes=table._attr_meta,
                                   determinants=table._determinants, formulas=table._formulas)
        synthetic._fitted = table._fitted
        synthetic._attributes = table._attributes
        # TODO: fk
        return synthetic

    def inverse_transform(self, normalized_core: Tensor, replace_content: bool = True) -> pd.DataFrame:
        if not self._fitted:
            raise NotFittedError('Table', 'inversely transforming predicted synthetic data')
        columns = {
            n: v.transformed_columns if n not in self._id_cols else [n]
            for n, v in self._attributes.items()
        }
        normalized_core = inverse_convert_data(normalized_core, pd.concat({
            n: pd.DataFrame(columns=v) for n, v in columns.items()
        }, axis=1).columns)[self._core_cols]

        recovered_df = pd.DataFrame()
        for col in self._core_cols:
            attribute = self._attributes[col]
            if col in self._id_cols:
                assert isinstance(attribute, SerialIDAttribute)
                recovered = attribute.generate(len(normalized_core))
            else:
                recovered = attribute.inverse_transform(normalized_core[col])
            recovered_df[col] = recovered

        for i, det in enumerate(self._determinants):
            leader, describer = det[0], self._describers[i]
            for grp_name, data in self._data.groupby(by=[leader], sort=False, dropna=False):
                if pd.isnull(grp_name):
                    grp_name = self._attributes[leader].fill_nan_val
                generator = DataGenerator()
                tempfile_name = f'{self._name}__{leader}__{grp_name}'
                with open(f'{tempfile_name}.json', 'w') as f:
                    json.dump(describer[grp_name], f)
                    f.close()
                generator.generate_dataset_in_correlated_attribute_mode(len(data), f'{tempfile_name}.json')
                generated: pd.DataFrame = generator.synthetic_dataset.drop(columns=[':dummy'])\
                    .set_axis(list(data.index))
                recovered_df.loc[data.index, det[1:]] = generated
                os.remove(f'{tempfile_name}.json')

        for col, formula in self._formulas:
            recovered_df[col] = recovered_df.apply(eval(formula), axis=1)

        recovered_df = recovered_df[[*self._attributes]]
        if replace_content:
            self._data = recovered_df
            self._normalized_by_attr = {
                n: pd.DataFrame(normalized_core[n], columns=v) for n, v in columns.items()
            }

        return recovered_df

    def assign_degrees(self, degrees: pd.Series):
        self._degree[('', 'degree')] = degrees
        self._degree_normalized_by_attr[('', 'degree')] = self._degree_attributes[('', 'degree')].transform(degrees)
        self._augmented = self._degree.loc[self._degree.index.repeat(self._degree[self._degree.index])]\
            .reset_index(drop=True)
        for (table, attr_name), attr in self._augmented_attributes.items():
            if (table, attr_name) in self._augmented_ids:
                if table != self._name or attr_name in self._known_cols:
                    self._augmented_normalized_by_attr[(table, attr_name)] = self._augmented[[(table, attr_name)]]


