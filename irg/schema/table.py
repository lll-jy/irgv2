"""Table data structure that holds data and metadata of tables in a database."""

import json
import os
from typing import Optional, Iterable, Dict, Tuple, Set, List
import pickle

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
    """Table data structure that holds metadata description of the table content and the relevant data."""
    def __init__(self, name: str, need_fit: bool = True, id_cols: Optional[Iterable[str]] = None,
                 attributes: Optional[Dict[str, dict]] = None, data: Optional[pd.DataFrame] = None,
                 determinants: Optional[List[List[str]]] = None, formulas: Optional[Dict[str, str]] = None, **kwargs):
        """
        **Args**:

        - `name` (`str`): Name of the table.
        - `need_fit` (`bool`): Whether the table need to be fitted. Default is `True`.
        - `id_cols` (`Optional[Iterable[str]]`): ID column names
        - `attributes` (`Optional[Dict[str, dict]]`): Attribute metadata readable by
          [`create_attribute`](attribute#irg.schema.attribute.create).
          If not provided, it will be inferred from `data`.
        - `data` (`Optional[pd.DataFrame]`): Data content of the table.
          It can be deferred. But attributes and data must not be both `None`.
        - `determinants` (`Optional[List[List[str]]]`): Determinant groups for BN generation.
        - `formulas` (`Optional[Dict[str, str]]`): Formula constraints of columns, provided as a dict of column name
          and a function (lambda expression permitted) processable by `eval` built-in function, with the only argument
          is a row in the table.
        - `kwargs`: Other arguments for `DataSynthesizer.DataDescriber` constructor.
        """
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
        self._augmented_attributes: Dict[TwoLevelName, BaseAttribute] = {}
        self._degree_attributes: Dict[TwoLevelName, BaseAttribute] = {}
        self._augmented_normalized_by_attr: Dict[TwoLevelName, pd.DataFrame] = {}
        self._degree_normalized_by_attr: Dict[TwoLevelName, pd.DataFrame] = {}
        self._augmented_ids: Set[TwoLevelName] = set()
        self._degree_ids: Set[TwoLevelName] = set()

    @property
    def name(self) -> str:
        """Name of the table."""
        return self._name

    @property
    def columns(self) -> List[str]:
        """Name of columns."""
        return [*self._attributes]

    @property
    def id_cols(self) -> Set[str]:
        """Set if columns that are ID."""
        return self._id_cols

    def augment(self, augmented: pd.DataFrame, degree: pd.DataFrame,
                augmented_ids: Set[TwoLevelName], degree_ids: Set[TwoLevelName],
                augmented_attributes: Dict[TwoLevelName, BaseAttribute],
                degree_attributes: Dict[TwoLevelName, BaseAttribute]):
        """
        Save augmented data.

        **Args**:

        - `augmented` (`pd.DataFrame`): Augmented data.
        - `degree`: (`pd.DataFrame`): Degree data.
        - `augmented_ids` (`Set[TwoLevelName]`): ID columns in augmented tables as a set of tuples.
        - `degree_ids` (`Set[TwoLevelName]`): ID columns in degree tables as a set of tuples.
        - `augmented_attributes` (`Dict[TwoLevelName, BaseAttribute]`): Attributes (typically fitted) of the augmented
          table.
        - `degree_attributes` (`Dict[TwoLevelName, BaseAttribute]`): Attributes (typically fitted) of the degree
          table.
        """
        self._augmented, self._degree = augmented, degree
        self._augmented_ids, self._degree_ids = augmented_ids, degree_ids
        self._augmented_attributes, self._degree_attributes = augmented_attributes, degree_attributes

        for name, attr in self._augmented_attributes.items():
            self._augmented_normalized_by_attr[name] = attr.transform(self._augmented[name])
        for name, attr in self._degree_attributes.items():
            self._degree_normalized_by_attr[name] = attr.transform(self._degree[name])
        self._augment_fitted = True

    def fit(self, data: pd.DataFrame, force_redo: bool = False, **kwargs):
        """
        Fit the table with given data.

        **Args**:

        - `data` (`pd.DataFrame`): The data content to fit the table.
        - `force_redo` (`bool`): Whether to re-fit the table if the table is already fitted. Default is `False`.
        - `kwargs`: Other arguments for `DataSynthesizer.DataDescriber` constructor.
        """
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
        """
        Get the specified table data content.

        **Args**:

        - `variant` (`str`): Choose one from 'original', 'augmented', 'degree'. Default is 'original'.
        - `normalize` (`bool`): Whether to return the normalized data. Default is `False`.
        - `with_id` (`str`): ID return policy, choose one from 'this' (IDs of this table only), 'none' (no ID columns),
          and 'inherit' (IDs from this and other tables by augmentation).
        - `core_only` (`bool`): Whether to return core columns only, or include determinant and formula columns.
          Default is `False`.
        - `return_as` (`str`): Return format as per [`convert_data_as`](../utils/misc#irg.utils.misc.convert_data_as).

        **Return**: The queried data of the desired format.

        **Raise**: `NotImplementedError` if variant and with_id policies are not recognized.
        """
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
        """Whether the table is independent (i.e. no parents)"""
        return not self._augment_fitted

    @staticmethod
    def _attr2catdim(attributes: Dict[str, BaseAttribute]) -> List[Tuple[int, int]]:
        base, res = 0, []
        for name, attr in attributes.items():
            res += attr.categorical_dimensions(base)
            base += len(attr.transformed_columns)
        return res

    @property
    def augmented_for_join(self) -> Tuple[pd.DataFrame, Set[str], Dict[str, BaseAttribute]]:
        """Augmented information for joining, including augmented table, set of ID column names, and attributes."""
        if self.is_independent:
            return self.data(), self._id_cols, self._attributes

        data = self.data(variant='augmented')
        flattened, attributes = {}, {}
        for (table, col), group_df in data.groupby(level=[0, 1]):
            col_name = col if table == self.name else f'{table}/{col}'
            attributes[col_name] = self._augmented_attributes[(table, col)]
            flattened[col_name] = group_df
        return pd.concat(flattened, axis=1), self._id_cols, attributes

    @property
    def ptg_data(self) -> Tuple[Tensor, Tensor, List[Tuple[int, int]]]:
        """Data used for tabular data generation (X, y) with a list showing
        [categorical columns](../tabular/ctgan#irg.tabular.ctgan.CTGANTrainer)."""
        if not self.is_independent:
            unknown_cols = [
                (table, attr) for table, attr in self._augmented_attributes
                if table == self._name and attr not in self._known_cols
            ]
            aug_data = self.data(variant='augmented', normalize=True, with_id='none', core_only=True)
            unknown_set = set(unknown_cols)
            known_cols = [col for col in aug_data.columns.droplevel(2) if col not in unknown_set]
            known_data, unknown_data = aug_data[known_cols], aug_data[unknown_cols]
            cat_dims = self._attr2catdim({
                table: attr for table, attr in self._augmented_attributes
                if table == self._name and attr not in self._known_cols
            })
            return convert_data_as(known_data, 'torch'), convert_data_as(unknown_data, 'torch'), cat_dims
        else:
            norm_data = self.data(variant='original', normalize=True, with_id='none', core_only=True)
            return torch.zeros(len(norm_data), 0), convert_data_as(norm_data, 'torch'), \
                   self._attr2catdim(self._attributes)

    @property
    def deg_data(self) -> Tuple[Tensor, Tensor, List[Tuple[int, int]]]:
        """Data used for degree generate (X, y) with a list showing
        [categorical columns](../tabular/ctgan#irg.tabular.ctgan.CTGANTrainer).
        Raises [`NoPartiallyKnownError`](../utils/errors#irg.utils.errors.NoPartiallyKnownError) if not independent."""
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
        cat_dims = self._attr2catdim({
            table: attr for table, attr in self._degree_attributes
            if table == self._name and attr not in self._known_cols
        })
        return convert_data_as(known_data, 'torch'), convert_data_as(unknown_data, 'torch'), cat_dims

    def save(self, path: str):
        """
        Save the table.

        - `path` (`str`): Path to save this table to.
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "Table":
        """
        Load table from path.

        - `path` (`str`): Path of the file to load.

        **Return**: Loaded table.
        """
        with open(path, 'rb') as f:
            loaded = pickle.load(f)
            loaded.__class__ = Table
        return loaded

    def __len__(self):
        return len(self._data)

    @property
    def attributes(self) -> Dict[str, BaseAttribute]:
        """All attributes of the table."""
        return self._attributes


class SyntheticTable(Table):
    """Synthetic counterpart of real tables."""
    @classmethod
    def from_real(cls, table: Table) -> "SyntheticTable":
        """
        Construct synthetic table from real one.

        **Args**:

        - `table` (`Table`): The original real table.

        **Return**: The constructed synthetic table.
        """
        synthetic = SyntheticTable(name=table._name, need_fit=False,
                                   id_cols={*table._id_cols}, attributes=table._attr_meta,
                                   determinants=table._determinants, formulas=table._formulas)
        synthetic._fitted = table._fitted
        synthetic._attributes = table._attributes
        return synthetic

    def inverse_transform(self, normalized_core: Tensor, replace_content: bool = True) -> pd.DataFrame:
        """
        Inversely transform normalized data to original data format.

        **Args**:

        - `normalized_core` (`torch.Tensor`): Normalized core data in terms of tensor.
        - `replace_content` (`bool`): Whether to replace the content of this `Table`. Default is `True`.

        **Return**: The inversely transformed data.

        **Raise**: [`NotFittedError`](../utils/errors#irg.utils.errors.NotFittedError) if the table is not yet fitted.
        """
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
        """
        Assign degrees to augmented table so the shape of the synthetically generated table is fixed.

        **Args**:

        - `degrees` (`pd.Series`): The degrees to be assigned.
        """
        self._degree[('', 'degree')] = degrees
        self._degree_normalized_by_attr[('', 'degree')] = self._degree_attributes[('', 'degree')].transform(degrees)
        self._augmented = self._degree.loc[self._degree.index.repeat(self._degree[self._degree.index])]\
            .reset_index(drop=True)
        for (table, attr_name), attr in self._augmented_attributes.items():
            if (table, attr_name) in self._augmented_ids:
                if table != self._name or attr_name in self._known_cols:
                    self._augmented_normalized_by_attr[(table, attr_name)] = self._augmented[[(table, attr_name)]]

    def inverse_transform_degrees(self, degree_tensor: Tensor, scale: float = 1) -> pd.Series:
        """
        Inversely transform degree predictions to a series of integers.

        **Args**:

        - `degree_tensor` (`torch.Tensor`): The raw prediction of degree as a tensor.
        - `scale` (`float`): Scaling factor of the generated degrees. Default is 1.

        **Return**: Recovered degrees.
        """
        degrees = self._degree_attributes[('', 'degree')].inverse_transform(degree_tensor)
        degrees = degrees.apply(lambda x: max(0, round(x * scale)))
        return degrees
