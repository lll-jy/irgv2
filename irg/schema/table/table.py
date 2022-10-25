"""Normal tabular table data structure that holds data and metadata of tables in a database."""

import json
import logging
import os
import pickle
import re
import shutil
from types import FunctionType
from typing import Optional, Iterable, Dict, Tuple, Set, List, ItemsView, Any, Literal, Union

import torch
from torch import Tensor
import pandas as pd
import numpy as np
from tqdm import tqdm
from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator

from ..attribute import learn_meta, create as create_attribute, BaseAttribute, SerialIDAttribute, RawAttribute
from ...utils.misc import Data2D, Data2DName, convert_data_as, inverse_convert_data
from ...utils.errors import NoPartiallyKnownError, NotFittedError
from ...utils.dist import fast_map_dict, fast_map
from ...utils.io import pd_to_pickle, pd_read_compressed_pickle, HiddenPrints

TwoLevelName = Tuple[str, str]
"""Two-level name type, which is a tuple of two strings."""
Variant = Literal['original', 'augmented', 'degree']
"""Table variant. Literal of `original`, `augmented`, or `degree`."""
IdPolicy = Literal['none', 'this', 'inherit']
"""
Data retrieval ID policy. Literal of `none`, `this`, `inherit`.

- `none`: no ID columns.
- `this`: IDs of this table only.
- `inherit` :IDs from this and other tables by augmentation.
"""
_LOGGER = logging.getLogger()


class Table:
    """Table data structure that holds metadata description of the table content and the relevant data."""
    def __init__(self, name: str, ttype: str = 'normal', need_fit: bool = True, id_cols: Optional[Iterable[str]] = None,
                 attributes: Optional[Dict[str, dict]] = None, data: Optional[pd.DataFrame] = None,
                 determinants: Optional[List[List[str]]] = None, formulas: Optional[Dict[str, str]] = None,
                 temp_cache: str = '.temp', **kwargs):
        """
        **Args**:

        - `name` (`str`): Name of the table.
        - `ttype` (`str`): Table type. Recognized types include (default is `normal`)
            - `normal`: General tabular data.
            - `base`: Basic tabular-form information that does not need to be trained and/or generated.
              For example, subjects in school are not in any sense sensitive data, and remain it as it is
              makes the generated data makes more sense. These tables will be skipped from training and generation
              and use the real table as the result directly.
            - `series`: Series data information.
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
        - `temp_cache` (`str`): Directory path to save cached temporary files. Default is `.temp`.
        - `kwargs`: Other arguments for `DataSynthesizer.DataDescriber` constructor.
        """
        self._name, self._ttype, self._need_fit, self._fitted = name, ttype, need_fit, False
        self._determinants = [] if determinants is None else determinants
        self._formulas = {} if formulas is None else formulas
        id_cols = set() if id_cols is None else set(id_cols)

        self._temp_cache = temp_cache
        os.makedirs(temp_cache, exist_ok=True)
        os.makedirs(os.path.join(self._temp_cache, 'normalized'), exist_ok=True)
        os.makedirs(os.path.join(self._temp_cache, 'describers'), exist_ok=True)
        os.makedirs(os.path.join(self._temp_cache, 'norm_aug'), exist_ok=True)
        os.makedirs(os.path.join(self._temp_cache, 'norm_deg'), exist_ok=True)
        os.makedirs(os.path.join(self._temp_cache, 'attributes'), exist_ok=True)

        if attributes is None:
            if data is None:
                raise ValueError('Data and attributes cannot both be `None` to create a table.')
            attributes = self.learn_meta(data, id_cols)
            _LOGGER.debug(f'Learned metadata for table {self._name}.')
        self._length = None
        self._attr_meta, self._id_cols = attributes, id_cols
        self._attributes = {}
        self._attributes: Dict[str, BaseAttribute] = fast_map_dict(
            func=self._create_attribute,
            dictionary=self._attr_meta,
            verbose_descr=f'Construct attribute for {self._name}',
            func_kwargs=dict(need_fit=need_fit, data=data)
        )

        det_child_cols = {col for det in self._determinants for col in det[1:]}
        self._core_cols = [
            col for col in self._attributes
            if col not in det_child_cols and col not in self._formulas
        ]

        _LOGGER.debug(f'Loaded required information for Table {name}.')
        if need_fit and data is not None:
            self.fit(data, **kwargs)

        self._known_cols, self._unknown_cols, self._augment_fitted = [], [*self._attributes.keys()], False
        self._augmented_attributes: Dict[TwoLevelName, BaseAttribute] = {}
        self._degree_attributes: Dict[TwoLevelName, BaseAttribute] = {}
        self._aug_norm_by_attr_files: Dict[TwoLevelName, str] = {}
        self._deg_norm_by_attr_files: Dict[TwoLevelName, str] = {}
        self._augmented_ids: Set[TwoLevelName] = set()
        self._degree_ids: Set[TwoLevelName] = set()

    @property
    def ttype(self) -> str:
        """Table type."""
        return self._ttype

    def shallow_copy(self) -> "Table":
        """
        Make a shallow copy of `Table`.

        **Return**: Copied table.
        """
        copied = Table(
            name=self._name, ttype=self._ttype, need_fit=False, attributes=self._attr_meta,
            id_cols=self._id_cols, determinants=self._determinants, formulas=self._formulas,
            temp_cache=self._temp_cache
        )
        attr_to_copy = [
            '_fitted', '_attributes', '_length',
            '_known_cols', '_unknown_cols', '_augment_fitted',
            '_augmented_attributes', '_degree_attributes',
            '_aug_norm_by_attr_files', '_deg_norm_by_attr_files',
            '_augmented_ids', '_degree_ids'
        ]
        for attr in attr_to_copy:
            setattr(copied, attr, getattr(self, attr))
        return copied

    def _create_attribute(self, attr_name: str, meta: Dict[str, Any],
                          need_fit: bool = True, data: Optional[pd.DataFrame] = None) -> BaseAttribute:
        res = create_attribute(
            meta=meta,
            values=data[attr_name] if need_fit and data is not None else None,
            temp_cache=self._attribute_cache_path(attr_name)
        )
        return res

    def update_temp_cache(self, new_path: str):
        """
        Update cache directory.

        **Args**:

        - `new_path` (`str`): Path of the new directory.
        """
        shutil.copytree(self._temp_cache, new_path, dirs_exist_ok=True)
        self._temp_cache = new_path

    def _data_path(self) -> str:
        return os.path.join(self._temp_cache, 'data.pkl')

    def _normalized_path(self, attr_name: str) -> str:
        attr_name = attr_name.replace('/', ':')
        return os.path.join(self._temp_cache, 'normalized', f'{attr_name}.pkl')

    def _describer_path(self, idx: int) -> str:
        return os.path.join(self._temp_cache, 'describers', f'describer{idx}.json')

    def _augmented_path(self) -> str:
        return os.path.join(self._temp_cache, 'aug.pkl')

    def _degree_path(self) -> str:
        return os.path.join(self._temp_cache, 'deg.pkl')

    def _attribute_cache_path(self, attr_name: str) -> str:
        return os.path.join(self._temp_cache, 'attributes', attr_name)

    @staticmethod
    def _reduce_name_level(two_level: TwoLevelName) -> str:
        left, right = two_level
        left = re.sub(f'[/:<>"|*^]', '&', left)
        right = re.sub(f'[/:<>"|*^]', '&', right)
        return f'{left}__{right}'

    def _augmented_normalized_path(self, attr_name: TwoLevelName) -> str:
        if attr_name not in self._aug_norm_by_attr_files:
            self._aug_norm_by_attr_files[attr_name] = self._reduce_name_level(attr_name)
        return os.path.join(self._temp_cache, 'norm_aug', f'{self._aug_norm_by_attr_files[attr_name]}.pkl')

    def _degree_normalized_path(self, attr_name: TwoLevelName) -> str:
        if attr_name not in self._deg_norm_by_attr_files:
            self._deg_norm_by_attr_files[attr_name] = self._reduce_name_level(attr_name)
        return os.path.join(self._temp_cache, 'norm_deg', f'{self._deg_norm_by_attr_files[attr_name]}.pkl')

    def _degree_attr_path(self) -> str:
        return os.path.join(self._temp_cache, 'deg_attr.pkl')

    @classmethod
    def learn_meta(cls, data: pd.DataFrame, id_cols: Optional[Iterable[str]] = None,
                   force_cat: Optional[Iterable[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Learn attribute meta input from data.

        **Args**:

        - `data` (`pd.DataFrame`): Source data of the table.
        - `id_cols` (`Optional[Iterable[str]]`): List/set of ID column names.
        - `force_cat` (`Optional[Iterable[str]]`): List/set of column names that are forced to be categorical
          (typically for categorical labels that are expressed as numbers).

        **Return**: Learned description of attributes metadata of the table.
        """
        id_cols = [] if id_cols is None else id_cols
        force_cat = [] if force_cat is None else force_cat
        return fast_map_dict(
            func=cls._learn_attr_meta,
            dictionary=data.to_dict('series'),
            func_kwargs=dict(id_cols=id_cols, force_cat=force_cat)
        )

    @staticmethod
    def _learn_attr_meta(attr_name: str, col_data: pd.Series, id_cols: Iterable[str], force_cat: Iterable[str]) \
            -> Dict[str, Any]:
        return learn_meta(col_data, attr_name in id_cols, attr_name) | \
               {'name': attr_name} if attr_name not in force_cat else {'name': attr_name, 'type': 'categorical'}

    def replace_data(self, new_data: pd.DataFrame, replace_attr: bool = True):
        """
        Replace the data content in the table.

        **Args**:

        - `new_data` (`pd.DataFrame`): New data to fill in the table.
        - `replace_attr` (`bool`): Whether to replace attribute content.
        """
        new_data.to_pickle(self._data_path())
        self._length = len(new_data)
        if replace_attr:
            fast_map_dict(
                func=self._replace_data_by_attr,
                dictionary=self._attributes,
                func_kwargs=dict(new_data=new_data)
            )

    def _replace_data_by_attr(self, n: Union[str, TwoLevelName], attr: BaseAttribute, new_data: pd.DataFrame,
                              variant: Variant = 'original') -> int:
        if n in new_data:
            transformed = attr.transform(new_data[n])
            path_by_variant = {
                'original': self._normalized_path,
                'augmented': self._augmented_normalized_path,
                'degree': self._degree_normalized_path
            }
            if isinstance(n, str):
                n = n.replace('/', ':')
            else:
                n = n[0].replace('/', ':'), n[1].replace('/', ':')
            pd_to_pickle(transformed, path_by_variant[variant](n))
        return 0

    def replace_attributes(self, new_attributes: Dict[str, BaseAttribute]):
        """
        Replace some attributes.

        **Args**:

        - `new_attributes` (`Dict[str, BaseAttribute]`): New attributes to replace the old ones.

        **Raises**: `KeyError` if some attribute names are not recognized.
        """
        attributes = [*self._attributes]
        if {*new_attributes} > set(attributes):
            raise KeyError('New attributes should be a subset of existing attribute names, but got unseen ones.')

        fast_map(
            func=self._replace_attr_by_attr,
            iterable=attributes,
            total_len=len(attributes),
            func_kwargs=dict(new_attributes=new_attributes),
            filter_input=self._check_attr_in_new,
            input_kwargs=dict(new_attributes=new_attributes)
        )
        for attr_name in attributes:
            if attr_name not in new_attributes:
                continue
            self._attributes[attr_name] = new_attributes[attr_name]
            data = pd.read_pickle(self._data_path())
            if attr_name in data.columns:
                new_transformed = new_attributes[attr_name].transform(data[attr_name])
                pd_to_pickle(new_transformed, self._normalized_path(attr_name))

    def _replace_attr_by_attr(self, attr_name: str, new_attributes: Dict[str, BaseAttribute]) -> int:
        self._attributes[attr_name] = new_attributes[attr_name]
        data = pd.read_pickle(self._data_path())
        if attr_name in data.columns:
            new_transformed = new_attributes[attr_name].transform(data[attr_name])
            pd_to_pickle(new_transformed, self._normalized_path(attr_name))
        return 0

    @staticmethod
    def _check_attr_in_new(attr_name: str, new_attributes: Dict[str, BaseAttribute]) -> bool:
        return attr_name in new_attributes

    def join(self, right: "Table", ref: ItemsView[str, str], descr: Optional[str] = None, how: str = 'outer') \
            -> "Table":
        """
        Join two tables.

        **Args**:

        - `right` (`Table`): The other table to be joined.
        - `ref` (`ItemsView[str, str]`): Items view of columns of this table matched to the other table for joining.
        - `descr` (`Optional[str]`): Name of the resulting table. If not provided, the naming will be
          `{THIS_NAME}_{HOW}_joined_{RIGHT_NAME}`.
        - `how` (`str`): How to join. Default is `outer`.

        **Return**: The joined table.
        """
        id_cols = {f'{self._name}/{col}' for col in self._id_cols} | {f'{right._name}/{col}' for col in right._id_cols}

        left_data, right_data = self.data(), right.data()
        left_data = left_data.rename(lambda x: f'{self._name}/{x}', axis=1)
        right_data = right_data.rename(lambda x: f'{right._name}/{x}', axis=1)

        left_on, right_on = [], []
        for left_col, right_col in ref:
            left_on.append(f'{self._name}/{left_col}')
            right_on.append(f'{right._name}/{right_col}')

        for left_col, right_col in zip(left_on, right_on):
            if left_data[left_col].dtype == 'O':
                right_data = right_data.astype({right_col: 'O'})
            elif right_data[right_col].dtype == 'O':
                left_data = left_data.astype({left_col: 'O'})
        joined = left_data.merge(right_data, how=how, left_on=left_on, right_on=right_on)

        os.makedirs(os.path.join(self._temp_cache, 'pair_joined'), exist_ok=True)
        descr = descr if descr is not None else f'{self._name}_{how}_join_{right._name}'
        result = Table(
            name=descr,
            need_fit=True, id_cols=id_cols, data=joined,
            temp_cache=os.path.join(self._temp_cache, 'pair_joined', descr)
        )
        result._fitted = True

        result._attr_meta = {}
        for n, v in self._attr_meta.items():
            content = v.copy()
            if 'name' in v:
                content['name'] = f'{self._name}/{content["name"]}'
            result._attr_meta[f'{self._name}/{n}'] = content
        for n, v in right._attr_meta.items():
            content = v.copy()
            if 'name' in v:
                content['name'] = f'{right._name}/{content["name"]}'
            result._attr_meta[f'{right._name}/{n}'] = content

        result._attributes = {}
        for n, v in self._attributes.items():
            result._attributes[f'{self._name}/{n}'] = v.rename(f'{self._name}/{v.name}', inplace=False)
        for n, v in right._attributes.items():
            result._attributes[f'{right._name}/{n}'] = v.rename(f'{right._name}/{v.name}', inplace=False)

        fast_map_dict(
            func=result._replace_data_by_attr,
            dictionary=result._attributes,
            func_kwargs=dict(new_data=joined)
        )

        _LOGGER.debug(f'Joined {self._name} with {right._name}.')

        return result

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
        self._known_cols = [col for (table, col) in degree_attributes if table == self._name]
        self._unknown_cols = [col for col in self._unknown_cols if col not in self._known_cols]
        if len(self._known_cols) > 0 and augmented_attributes:
            groupby_cols = [(self._name, col) for col in self._known_cols]
            sizes = degree.loc[:, groupby_cols].groupby(groupby_cols).size()
            degree = degree.merge(pd.DataFrame({('', 'degree'): sizes}), on=groupby_cols)

        augmented.to_pickle(self._augmented_path())
        degree.to_pickle(self._degree_path())
        self._augmented_ids, self._degree_ids = augmented_ids, degree_ids
        self._augmented_attributes, self._degree_attributes = augmented_attributes, degree_attributes
        if len(self._known_cols) > 0:
            deg_meta = {
                'type': 'numerical',
                'rounding': 0,
                'min_val': 0,
                'name': 'degree'
            }
            self._degree_attributes[('', 'degree')] = create_attribute(
                deg_meta,
                values=degree.loc[:, ('', 'degree')],
                temp_cache=self._degree_attr_path()
            )
            print('done create', flush=True)
        print('done attributes', flush=True)

        fast_map_dict(
            func=self._replace_data_by_attr,
            dictionary=self._augmented_attributes,
            func_kwargs=dict(new_data=augmented, variant='augmented'),
            verbose_descr=f'Replacing augmented {self._name}'
        )
        fast_map_dict(
            func=self._replace_data_by_attr,
            dictionary=self._degree_attributes,
            func_kwargs=dict(new_data=degree, variant='degree'),
            verbose_descr=f'Replacing degree {self._name}'
        )
        print('updated attribute', flush=True)
        _LOGGER.debug(f'Augmented {self._name} has columns {augmented.columns.values}. '
                      f'The augmented table has {len(augmented)} rows, and degree table has {len(degree)} rows.')
        self._augment_fitted = True
        self._fitted = True
        if self._length is None:
            self._length = len(augmented)

    def fit(self, data: pd.DataFrame, force_redo: bool = False, **kwargs):
        """
        Fit the table with given data.

        **Args**:

        - `data` (`pd.DataFrame`): The data content to fit the table.
        - `force_redo` (`bool`): Whether to re-fit the table if the table is already fitted. Default is `False`.
        - `kwargs`: Other arguments for `DataSynthesizer.DataDescriber` constructor.
        """
        if (self._fitted and not force_redo) or not self._need_fit:
            _LOGGER.info(f'Table {self._name} is already fitted. Duplicated fitting is skipped.')
            return
        self._length = len(data)
        data = data[[*self._attributes.keys()]]
        data.to_pickle(self._data_path())
        fast_map_dict(
            func=self._fit_attribute,
            dictionary=self._attributes,
            verbose_descr=f'Fit attributes for {self._name}',
            func_kwargs=dict(data=data, force_redo=force_redo)
        )

        _LOGGER.debug(f'Fitted attributes for Table {self._name}.')

        if self._ttype != 'base':
            with HiddenPrints():
                map_dict_base = tqdm(dict(zip(range(len(self._determinants)), self._determinants)).items(),
                                     desc=f'Fit determinants for {self._name}')
                for k, v in map_dict_base:
                    self._fit_determinant_helper(k, v, data=data, **kwargs)
            _LOGGER.debug(f'Fitted determinants for Table {self._name}.')

        self._fitted = True
        _LOGGER.info(f'Fitted Table {self._name}.')

    def _fit_attribute(self, name: str, attr: BaseAttribute, data: pd.DataFrame, force_redo: bool) -> int:
        attr.fit(data[name], force_redo=force_redo)
        pd_to_pickle(attr.get_original_transformed(), self._normalized_path(name))
        return 0

    def _fit_determinant_helper(self, i: int, det: List[str], data: pd.DataFrame, **kwargs) -> int:
        for col in det:
            if self._attributes[col].atype not in {'categorical', 'id'}:
                raise TypeError('Determinant should have all columns categorical (or rarely, ID).')
        leader, children = det[0], det[1:]
        det_describers = self._fit_determinant(data, leader, children, **kwargs)
        with open(self._describer_path(i), 'w') as f:
            json.dump(det_describers, f)
        return 0

    def _fit_determinant(self, complete_data: pd.DataFrame, leader: str, children: List[str], **kwargs) \
            -> Dict[str, Dict]:
        describers = dict()
        os.makedirs(os.path.join(self._temp_cache, 'temp_det'), exist_ok=True)
        for grp_name, data in complete_data[[leader] + children]\
                .groupby(by=[leader], sort=False, dropna=False):
            describer = DataDescriber(**kwargs)
            if len(data) <= 1:
                data = pd.concat([data, data]).reset_index(drop=True)
            if pd.isnull(grp_name):
                grp_name = self._attributes[leader].fill_nan_val
            data = data.copy()
            data[leader] = str(grp_name)
            for col in children:
                data[col] = data[col].fillna(self._attributes[col].fill_nan_val)
            data[':dummy'] = 'dummy'
            tempfile_name = os.path.join(self._temp_cache, 'temp_det', f'{self._name}__{leader}__{grp_name}')
            data = data.applymap(lambda x: f'c{x}', na_action='ignore')
            data.to_csv(f'{tempfile_name}.csv', index=False)
            describer.describe_dataset_in_correlated_attribute_mode(
                f'{tempfile_name}.csv', k=len(children)+2, epsilon=0,
                attribute_to_datatype={col: 'String' for col in children} | {leader: 'String'} | {':dummy': 'String'},
                attribute_to_is_categorical={col: True for col in children} | {leader: True} | {':dummy': True},
                attribute_to_is_candidate_key={col: False for col in children} | {leader: False} | {':dummy': False}
            )
            describer.save_dataset_description_to_file(f'{tempfile_name}.json')
            with open(f'{tempfile_name}.json', 'r') as f:
                describer_info = json.load(f)
                f.close()
            os.remove(f'{tempfile_name}.csv')
            os.remove(f'{tempfile_name}.json')
            describers[grp_name] = describer_info
        os.removedirs(os.path.join(self._temp_cache, 'temp_det'))
        return describers

    def transform(self, data: Data2D, with_id: IdPolicy = 'this', core_only: bool = False, return_as: Data2DName = 'pandas') -> Data2D:
        exclude_cols = self._id_cols if with_id == 'none' else set()
        exclude_cols |= {col for col in self._attributes if col not in self._core_cols} if core_only else set()
        data = pd.concat({
            n: attr.transform(data[n], return_as)
            for n, attr in self._attributes.items() if n not in exclude_cols
        }, axis=1)
        return convert_data_as(data, return_as)

    def data(self, variant: Variant = 'original', normalize: bool = False,
             with_id: IdPolicy = 'this', core_only: bool = False, return_as: Data2DName = 'pandas') -> Data2D:
        """
        Get the specified table data content.

        **Args**:

        - `variant` (`Variant`): Choose one from 'original', 'augmented', 'degree'. Default is 'original'.
        - `normalize` (`bool`): Whether to return the normalized data. Default is `False`.
        - `with_id` (`IdPolicy`): ID return policy.
        - `core_only` (`bool`): Whether to return core columns only, or include determinant and formula columns.
          Default is `False`.
        - `return_as` (`Data2DName`): Return format as per [`convert_data_as`](../utils/misc#irg.utils.misc.convert_data_as).

        **Return**: The queried data of the desired format.

        **Raise**: `NotImplementedError` if variant and with_id policies are not recognized.
        """
        if with_id not in {'this', 'none', 'inherit'}:
            raise NotImplementedError(f'With id policy "{with_id}" is not recognized.')
        if self._length is None:
            raise NotFittedError('Table', 'getting its data')
        if variant == 'original':
            data = pd.read_pickle(self._data_path())
            exclude_cols = self._id_cols if with_id == 'none' else set()
            exclude_cols |= {col for col in self._attributes if col not in self._core_cols} if core_only else set()
            if not normalize:
                data = data[[col for col in data.columns if col not in exclude_cols]]
            else:
                data = pd.concat({
                    n: pd_read_compressed_pickle(self._normalized_path(n))
                    for n, attr in self._attributes.items() if n not in exclude_cols
                }, axis=1)
        elif variant == 'augmented':
            augmented = pd.read_pickle(self._augmented_path())
            data = self._get_aug_or_deg_data(augmented, self._aug_norm_by_attr_files,
                                             self._augmented_ids, normalize, with_id, core_only,
                                             self._augmented_normalized_path)
        elif variant == 'degree':
            degree = pd.read_pickle(self._degree_path())
            data = self._get_aug_or_deg_data(degree, self._deg_norm_by_attr_files, self._degree_ids,
                                             normalize, with_id, core_only, self._degree_normalized_path)
        else:
            raise NotImplementedError(f'Getting data variant "{variant}" is not recognized.')

        return convert_data_as(data, return_as=return_as, copy=True)

    def _get_aug_or_deg_data(self, data: pd.DataFrame, normalized_by_attr: Dict[TwoLevelName, str],
                             id_cols: Set[TwoLevelName], normalize: bool = False, with_id: IdPolicy = 'this',
                             core_only: bool = False, path_getter: Optional[FunctionType] = None) -> pd.DataFrame:
        if self.is_independent():
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
            if not normalized_by_attr:
                raise ValueError()
            to_concat = {
                n: pd_read_compressed_pickle(path_getter(n)) for n
                in normalized_by_attr if n not in exclude_cols
            }
            data = pd.concat(to_concat, axis=1) if to_concat else pd.DataFrame()
            if data.empty:
                raise ValueError('hello this is empty', self._name, exclude_cols, [*normalized_by_attr])
        return data

    def is_independent(self):
        """Whether the table is independent (i.e. no parents)"""
        return not self._augment_fitted or len(self._known_cols) == 0

    @staticmethod
    def _attr2catdim(attributes: Dict[str, BaseAttribute]) -> List[Tuple[int, int]]:
        base, res = 0, []
        for name, attr in attributes.items():
            res += attr.categorical_dimensions(base)
            base += len(attr.transformed_columns)
        return res

    def augmented_for_join(self) -> Tuple[pd.DataFrame, Set[str], Dict[str, BaseAttribute]]:
        """
        Get the augmented information for joining.

        **Args**:

        - `normalized` (`bool`): Whether to return the normalized. Default is `False`.

        **Return**: Augmented table, set of ID column names, and attributes.
        """
        if self.is_independent():
            return self.data(), self._id_cols, self._attributes

        data = self.data(variant='augmented')
        flattened, attributes = {}, {n: v for n, v in self._attributes.items()}
        for (table, col), group_df in data.groupby(level=[0, 1], axis=1):
            if table == '':
                continue
            col_name = col if table == self.name else f'{table}/{col}'
            attributes[col_name] = self._augmented_attributes[(table, col)]
            flattened[col_name] = group_df[(table, col)]
        return pd.concat(flattened, axis=1), self._id_cols, attributes

    def ptg_data(self) -> Tuple[Tensor, Tensor, List[Tuple[int, int]]]:
        """Data used for tabular data generation (X, y) with a list showing
        [categorical columns](../tabular/ctgan#irg.tabular.ctgan.CTGANTrainer)."""
        if not self.is_independent():
            unknown_cols = [
                (table, attr) for table, attr in self._augmented_attributes
                if table == self._name and attr not in self._known_cols
            ]
            aug_data = self.data(variant='augmented', normalize=True, with_id='inherit', core_only=True)
            unknown_set = set(unknown_cols)
            known_cols = [col for col in aug_data.columns.droplevel(2) if col not in unknown_set]
            known_data = aug_data[[(a, b, c) for a, b, c in aug_data.columns if (a, b) in known_cols]]
            unknown_data = aug_data[[(a, b, c) for a, b, c in aug_data.columns if (a, b) in unknown_cols]]
            unknown_attr = {
                (table, attr_name): attr for (table, attr_name), attr in self._augmented_attributes.items()
                if table == self._name and attr_name not in self._known_cols
            }
            cat_dims = self._attr2catdim(unknown_attr)
            return convert_data_as(known_data, 'torch'), convert_data_as(unknown_data, 'torch'), cat_dims
        else:
            norm_data = self.data(variant='original', normalize=True, with_id='inherit', core_only=True)
            return (torch.zeros(len(norm_data), 0), convert_data_as(norm_data, 'torch'),
                    self._attr2catdim(self._attributes))

    def deg_data(self) -> Tuple[Tensor, Tensor, List[Tuple[int, int]]]:
        """Data used for degree generate (X, y) with a list showing
        [categorical columns](../tabular/ctgan#irg.tabular.ctgan.CTGANTrainer).
        Raises [`NoPartiallyKnownError`](../utils/errors#irg.utils.errors.NoPartiallyKnownError) if not independent."""
        if self.is_independent():
            raise NoPartiallyKnownError(self._name)
        unknown_cols = [('', 'degree')] + [
            (table, attr) for table, attr in self._degree_attributes
            if table == self._name and attr not in self._known_cols
        ]
        deg_data = self.data(variant='degree', normalize=True, with_id='none', core_only=True)
        unknown_set = set(unknown_cols)
        known_cols = [col for col in deg_data.columns.droplevel(2) if col not in unknown_set]
        known_data = deg_data[[(a, b, c) for a, b, c in deg_data.columns if (a, b) in known_cols]]
        unknown_data = deg_data[[(a, b, c) for a, b, c in deg_data.columns if (a, b) in unknown_cols]]
        cat_dims = self._degree_attributes[('', 'degree')].categorical_dimensions()
        return convert_data_as(known_data, 'torch'), convert_data_as(unknown_data, 'torch'), cat_dims

    def save(self, path: str):
        """
        Save the table. This will still rely on content of the temporary cache.

        **Args**:

        - `path` (`str`): Path to save this table to.
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def save_complete(self, dir_path: str):
        """
        Save complete information of the table. The result can be recognized independent of the temporary cache.

        **Args**:

        - `dir_path` (`str`): Path of the directory to save this table to.
        """
        os.makedirs(dir_path, exist_ok=True)
        self.save(os.path.join(dir_path, 'table_info.pkl'))
        shutil.copytree(self._temp_cache, dir_path, dirs_exist_ok=True)

    @classmethod
    def load_complete(cls, dir_path: str) -> "Table":
        """
        Load table from path. Inverse of `save_complete`.

        **Args:**

        - `dir_path` (`str`): Path of the directory to load.

        **Return**: Loaded table.
        """
        loaded = cls.load(os.path.join(dir_path, 'table_info.pkl'))
        shutil.copytree(dir_path, loaded._temp_cache, dirs_exist_ok=True)
        return loaded

    @classmethod
    def load(cls, path: str) -> "Table":
        """
        Load table from path. Inverse of `save`.

        **Args**:

        - `path` (`str`): Path of the file to load.

        **Return**: Loaded table.
        """
        with open(path, 'rb') as f:
            loaded = pickle.load(f)
            loaded.__class__ = Table
        return loaded

    def __len__(self):
        return self._length

    def attributes(self) -> Dict[str, BaseAttribute]:
        """All attributes of the table."""
        return self._attributes


class SyntheticTable(Table):
    """Synthetic counterpart of real tables."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._real_cache = '.temp' if 'temp_cache' not in kwargs else kwargs['temp_cache']

    @classmethod
    def load(cls, path: str) -> "SyntheticTable":
        with open(path, 'rb') as f:
            loaded = pickle.load(f)
            loaded.__class__ = SyntheticTable
        return loaded

    def _describer_path(self, idx: int) -> str:
        return os.path.join(self._real_cache, 'describers', f'describer{idx}.json')

    def _degree_attr_path(self) -> str:
        return os.path.join(self._real_cache, 'deg.pkl')

    @classmethod
    def from_real(cls, table: Table, temp_cache: Optional[str] = None) -> "SyntheticTable":
        """
        Construct synthetic table from real one.

        **Args**:

        - `table` (`Table`): The original real table.
        - `temp_cache` (`Optional[str]`): Directory path to save cached temporary files.
          If not provided, the real one will be taken.

        **Return**: The constructed synthetic table.
        """
        print('before')
        synthetic = SyntheticTable(name=table._name, ttype=table._ttype, need_fit=False,
                                   id_cols={*table._id_cols}, attributes=table._attr_meta,
                                   determinants=table._determinants, formulas=table._formulas,
                                   temp_cache=temp_cache if temp_cache is not None else table._temp_cache)
        print('end, then copy', flush=True)
        synthetic._fitted, synthetic._augment_fitted = table._fitted, table._augment_fitted
        synthetic._attributes = table._attributes
        synthetic._real_cache = table._temp_cache
        synthetic._known_cols, synthetic._unknown_cols = table._known_cols, table._unknown_cols
        synthetic._augmented_attributes = table._augmented_attributes
        synthetic._degree_attributes = table._degree_attributes
        synthetic._augmented_ids, synthetic._degree_ids = table._augmented_ids, table._degree_ids
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
            n: v.transformed_columns
            for n, v in self._attributes.items()
        }
        normalized_core = inverse_convert_data(normalized_core, pd.concat({
            n: pd.DataFrame(columns=v) for n, v in columns.items()
            if n in self._core_cols and n not in self._known_cols
        }, axis=1).columns)
        if not self.is_independent():
            augmented_df = pd.read_pickle(self._augmented_path())
        else:
            augmented_df = pd.DataFrame()

        recovered_df = pd.DataFrame()
        for col in self._core_cols:
            attribute = self._attributes[col]
            if col in normalized_core:
                print('directly inverse', col, self._name)
                recovered = attribute.inverse_transform(normalized_core[col])
            elif col in self._known_cols:
                print('directly get from augmented', col, self._name)
                recovered = augmented_df[(self._name, col)]
            else:
                print('generate', col, self._name)
                assert isinstance(attribute, SerialIDAttribute), f'Column cannot be recovered directly must be IDs. ' \
                                                                 f'Got {type(attribute)}.'
                recovered = attribute.generate(len(normalized_core))
            recovered_df[col] = recovered

        for x in self.id_cols:
            if x in self._unknown_cols:
                recovered_df[x] = self._attributes[x].generate(len(normalized_core))

        os.makedirs(os.path.join(self._temp_cache, 'temp_det'), exist_ok=True)
        for i, det in enumerate(self._determinants):
            leader = det[0]
            with open(self._describer_path(i), 'r') as f:
                describer = json.load(f)
            for grp_name, data in recovered_df.groupby(by=[leader], sort=False, dropna=False):
                if pd.isnull(grp_name):
                    grp_name = self._attributes[leader].fill_nan_val
                grp_name = str(grp_name)
                generator = DataGenerator()
                tempfile_name = os.path.join(self._temp_cache, 'temp_det', f'{self._name}__{leader}__{grp_name}')
                with open(f'{tempfile_name}.json', 'w') as f:
                    json.dump(describer[grp_name], f)
                    f.close()
                generator.generate_dataset_in_correlated_attribute_mode(len(data), f'{tempfile_name}.json')
                generated: pd.DataFrame = generator.synthetic_dataset.drop(columns=[':dummy'])\
                    .set_axis(list(data.index)).applymap(lambda x: x[1:], na_action='ignore')
                for col in data.columns:
                    nan_val = self._attributes[col].fill_nan_val
                    data.loc[data[col] == nan_val, col] = np.nan
                recovered_df.loc[data.index, det[1:]] = generated
                os.remove(f'{tempfile_name}.json')
        if os.path.exists(os.path.join(self._temp_cache, 'temp_det')):
            shutil.rmtree(os.path.join(self._temp_cache, 'temp_det'))

        for col, formula in self._formulas:
            recovered_df[col] = recovered_df.apply(eval(formula), axis=1)

        recovered_df = recovered_df[[*self._attributes]]
        if replace_content:
            recovered_df.to_pickle(self._data_path())
            for n, v in columns.items():
                if n in normalized_core:
                    pd_to_pickle(pd.DataFrame(normalized_core[n], columns=v), self._normalized_path(n))
                else:
                    pd_to_pickle(pd.DataFrame(
                        self._attributes[n].transform(recovered_df[n]), columns=v),
                        self._normalized_path(n)
                    )
            self._length = len(recovered_df)

        if 'student_token' in recovered_df:
            print('recovered student token:')
            print(recovered_df['student_token'].head())
        return recovered_df

    def assign_degrees(self, degrees: pd.Series):
        """
        Assign degrees to augmented table so the shape of the synthetically generated table is fixed.

        **Args**:

        - `degrees` (`pd.Series`): The degrees to be assigned.
        """
        degree_df = pd.read_pickle(self._degree_path())
        print('======== assigned degrees has degree known', degree_df.columns)
        print(degree_df.head())
        degree_df[('', 'degree')] = degrees
        pd_to_pickle(
            self._degree_attributes[('', 'degree')].transform(degrees),
            self._degree_normalized_path(('', 'degree'))
        )
        augmented = degree_df.loc[degree_df.index.repeat(degree_df[('', 'degree')]
                                                         .apply(lambda x: x if x >= 0 else 0))]\
            .reset_index(drop=True)
        augmented.to_pickle(self._augmented_path())
        for (table, attr_name), attr in self._augmented_attributes.items():
            if (table, attr_name) not in augmented.columns:
                continue
            transformed = attr.transform(augmented[(table, attr_name)])
            pd_to_pickle(transformed, self._augmented_normalized_path((table, attr_name)))
        self._fitted = True
        self._length = len(augmented)

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

    def update_deg_and_aug(self):
        """
        Update degree and augmented table after a new table is generated.
        """
        degree_df = pd.read_pickle(self._degree_path())
        augmented_df = pd.read_pickle(self._augmented_path())
        data = pd.concat({self._name: self.data()}, axis=1)
        print('^^ self data', data.columns.tolist())
        print('^^ gere deg', degree_df.columns.tolist())
        print('^^ here aug', degree_df.columns.tolist())
        degree_df = data.merge(degree_df, how='outer')
        augmented_df = data.merge(augmented_df.drop(columns=[('', 'degree')]), how='left')
        degree_df.to_pickle(self._degree_path())
        augmented_df.to_pickle(self._augmented_path())
        print('&&&&& updated aug', self._name, augmented_df.columns.tolist())
