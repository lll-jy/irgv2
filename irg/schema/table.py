from typing import Optional, Iterable, Dict, Tuple, Set

import pandas as pd

from .attribute import learn_meta, create as create_attribute, BaseAttribute, SerialIDAttribute
from ..utils.misc import Data2D, convert_data_as, inverse_convert_data
from ..utils.errors import NoPartiallyKnownError, NotFittedError

TwoLevelName = Tuple[str, str]


class Table:
    def __init__(self, name: str, need_fit: bool = True, id_cols: Iterable[str] = None,
                 attributes: Optional[Dict[str, dict]] = None, data: Optional[pd.DataFrame] = None):
        self._name, self._need_fit = name, need_fit
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
        self._normalized_by_attr = {}
        self._fitted = need_fit and data is not None
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

    def fit(self, data: pd.DataFrame, force_redo: bool = False):
        if (self._fitted and not force_redo) or not self._need_fit:
            return
        self._data = data
        for name, attr in self._attributes.items():
            attr.fit(data[name], force_redo=force_redo)
            self._normalized_by_attr[name] = attr.get_original_transformed()
        self._fitted = True

    def data(self, variant: str = 'original', normalize: bool = False,
             with_id: str = 'this', return_as: str = 'pandas') -> Data2D:
        if with_id not in {'this', 'none', 'inherit'}:
            raise NotImplementedError(f'With id policy "{with_id}" is not recognized.')
        if self._data is None:
            raise NotFittedError('Table', 'getting its data')
        if variant == 'original':
            exclude_cols = self._id_cols if with_id == 'none' else set()
            if not normalize:
                data = self._data[[col for col in self._data.columns if col not in exclude_cols]]
            else:
                data = pd.concat({n: v for n, v in self._normalized_by_attr.items() if n not in exclude_cols}, axis=1)
        elif variant == 'augmented':
            data = self._get_aug_or_deg_data(self._augmented, self._augmented_normalized_by_attr,
                                             self._augmented_ids, normalize, with_id)
        elif variant == 'degree':
            data = self._get_aug_or_deg_data(self._degree, self._degree_normalized_by_attr, self._degree_ids,
                                             normalize, with_id)
        else:
            raise NotImplementedError(f'Getting data variant "{variant}" is not recognized.')

        return convert_data_as(data, return_as=return_as, copy=True)

    def _get_aug_or_deg_data(self, data: pd.DataFrame, normalized_by_attr: Dict[TwoLevelName, pd.DataFrame],
                             id_cols: Set[TwoLevelName], normalize: bool = False, with_id: str = 'this') -> \
            pd.DataFrame:
        if self.is_independent:
            raise NoPartiallyKnownError(self._name)
        if with_id == 'inherit':
            exclude_cols = set()
        elif with_id == 'this':
            exclude_cols = {(table, attr) for table, attr in id_cols if table != self._name}
        else:
            assert with_id == 'none'
            exclude_cols = id_cols
        if not normalize:
            data = data[[col for col in data.columns if col not in exclude_cols]]
        else:
            data = pd.concat({n: v for n, v in normalized_by_attr.items() if n not in exclude_cols}, axis=1)
        return data

    @property
    def is_independent(self):
        return not self._augment_fitted

    @property
    def ptg_data(self) -> Tuple["Table", "Table"]:
        if self.is_independent:
            raise NoPartiallyKnownError(self._name)
        unknown_cols = {
            (table, attr) for table, attr in self._augmented_attributes
            if table == self._name and attr not in self._known_cols
        }
        return self._separate(
            data=self._augmented,
            unknown_cols=unknown_cols,
            normalized_by_attr=self._augmented_normalized_by_attr,
            id_cols=self._augmented_ids,
            attributes=self._augmented_attributes,
            attr_meta=self._augmented_meta
        )

    @property
    def deg_data(self) -> Tuple["Table", "Table"]:
        if self.is_independent:
            raise NoPartiallyKnownError(self._name)
        unknown_cols = {
            (table, attr) for table, attr in self._degree_attributes
            if (table == self._name and attr not in self._known_cols) or table == ''
        }
        return self._separate(
            data=self._degree,
            unknown_cols=unknown_cols,
            normalized_by_attr=self._degree_normalized_by_attr,
            id_cols=self._degree_ids,
            attributes=self._degree_attributes,
            attr_meta=self._degree_meta
        )

    def _separate(self, data: pd.DataFrame, unknown_cols: Set[TwoLevelName],
                  normalized_by_attr: Dict[TwoLevelName, pd.DataFrame], id_cols: Set[TwoLevelName],
                  attributes: Dict[TwoLevelName, BaseAttribute], attr_meta: Dict[TwoLevelName, dict]) \
            -> Tuple["Table", "Table"]:
        known_cols = {col for col in attributes if col not in unknown_cols}
        unknown = self._aug_or_deg_sub_table_from(
            data=data,
            columns=unknown_cols,
            normalized_by_attr=normalized_by_attr,
            id_cols=id_cols,
            attributes=attributes,
            attr_meta=attr_meta
        )
        known = self._aug_or_deg_sub_table_from(
            data=data,
            columns=known_cols,
            normalized_by_attr=normalized_by_attr,
            id_cols=id_cols,
            attributes=attributes,
            attr_meta=attr_meta
        )
        return known, unknown

    def _aug_or_deg_sub_table_from(self, data: pd.DataFrame, columns: Set[TwoLevelName],
                                   normalized_by_attr: Dict[TwoLevelName, pd.DataFrame], id_cols: Set[TwoLevelName],
                                   attributes: Dict[TwoLevelName, BaseAttribute], attr_meta: Dict[TwoLevelName, dict]) \
            -> "Table":
        new_table = Table(
            name=self._name,
            need_fit=False,
            id_cols={col for col in id_cols if col in columns},
            attributes={n: v for n, v in attr_meta.items() if n in columns}
        )
        new_table._data = data[columns]
        new_table._attributes = {n: v for n, v in attributes.items() if n in columns}
        new_table._normalized_by_attr = {n: v for n, v in normalized_by_attr.items() if n in columns}
        new_table._fitted = True
        return new_table


class SyntheticTable(Table):
    @classmethod
    def from_real(cls, table: Table) -> "SyntheticTable":
        synthetic = SyntheticTable(name=table._name, need_fit=False,
                                   id_cols={*table._id_cols}, attributes=table._attr_meta)
        synthetic._fitted = table._fitted
        synthetic._attributes = table._attributes
        # TODO: fk
        return synthetic

    def inverse_transform(self, normalized: Data2D):
        if not self._fitted:
            raise NotFittedError('Table', 'inversely transforming predicted synthetic data')
        columns = {
            n: v.transformed_columns if n not in self._id_cols else [n]
            for n, v in self._attributes.items()
        }
        normalized = inverse_convert_data(normalized, pd.concat({
            n: pd.DataFrame(columns=v) for n, v in columns.items()
        }, axis=1).columns)
        if not self.is_independent:
            normalized = normalized.set_axis(self._data)
        for col in self._unknown_cols:
            attribute = self._attributes[col]
            if col in self._id_cols:
                assert isinstance(attribute, SerialIDAttribute)
                recovered = attribute.generate(len(normalized))
            else:
                recovered = attribute.inverse_transform(normalized[col])
            if not self.is_independent:
                recovered = recovered.set_axis(self._data)
            self._data[col] = recovered

    def assign_degrees(self, degrees: pd.Series):
        self._degree[('', 'degree')] = degrees
        self._degree_normalized_by_attr[('', 'degree')] = self._degree_attributes[('', 'degree')].transform(degrees)
        self._augmented = self._degree.loc[self._degree.index.repeat(self._degree[self._degree.index])]\
            .reset_index(drop=True)
        for (table, attr_name), attr in self._augmented_attributes.items():
            if (table, attr_name) in self._augmented_ids:
                if table != self._name or attr_name in self._known_cols:
                    self._augmented_normalized_by_attr[(table, attr_name)] = self._augmented[[(table, attr_name)]]


