from typing import Optional, Iterable, Dict, Tuple, Set

import pandas as pd

from .attribute import learn_meta, create as create_attribute, BaseAttribute
from ..utils.misc import Data2D, convert_data_as

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
                attr_name: learn_meta(data[attr_name], attr_name in id_cols, attr_name)
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
                attr_name: attr.get_original_transformed()
                for attr_name, attr in self._attributes.items()
            }

        self._known_cols, self._unknown_cols = [], []
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

    def data(self, variant: str = 'original', normalize: bool = False,
             with_id: str = 'this', return_as: str = 'pandas') -> Data2D:
        if with_id not in {'this', 'none', 'inherit'}:
            raise NotImplementedError(f'With id policy "{with_id}" is not recognized.')
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

    def _get_aug_or_deg_data(self, data: pd.DataFrame, normalized_by_attr: Dict[TwoLevelName],
                             id_cols: Set[TwoLevelName], normalize: bool = False, with_id: str = 'this') -> \
            pd.DataFrame:
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
