from dateutil import parser
from typing import Optional, Dict

from jsonschema import validate
import pandas as pd

from .base import BaseAttribute
from .serial_id import SerialIDAttribute
from .categorical import CategoricalAttribute
from .numerical import NumericalAttribute
from .datetime import DatetimeAttribute, TimedeltaAttribute
from .encoding import EncodingAttribute

__all__ = (
    'BaseAttribute',
    'SerialIDAttribute',
    'CategoricalAttribute',
    'NumericalAttribute',
    'DatetimeAttribute',
    'TimedeltaAttribute',
    'EncodingAttribute',
    'create',
    'learn_meta'
)


_ATTR_CONF = {
    'type': 'object',
    'properties': {
        'name': {'type': 'string'},
        'type': {'enum': [
            'id',
            'categorical',
            'numerical',
            'datetime',
            'timedelta',
            'encoding'
        ]}
    },
    'required': ['name', 'type']
}
_ID_ATTR_CONF = {
    'type': 'object',
    'properties': {
        'name': {'type': 'string'},
        'type': {'enum': ['id']},
        'generator': {
            'type': 'string',
            'pattern': r'^\s*lambda\s+[a-zA-Z_][a-zA-Z0-9_]*\s*:'
        }
    },
    'required': ['name', 'type'],
    'additionalProperties': False
}
_CAT_ATTR_CONF = {
    'type': 'object',
    'properties': {
        'name': {'type': 'string'},
        'type': {'enum': ['categorical']},
    },
    'required': ['name', 'type'],
    'additionalProperties': False
}
_NUM_ATTR_CONF = {
    'type': 'object',
    'properties': {
        'name': {'type': 'string'},
        'type': {'enum': ['numerical']},
        'rounding': {'type': 'integer'},
        'min_val': {},
        'max_val': {},
        'max_clusters': {'type': 'integer'},
        'std_multiplier': {'type': 'integer'},
        'weight_threshold': {'type': 'number'}
    },
    'required': ['name', 'type'],
    'additionalProperties': False
}
_DT_ATTR_CONF = {
    'type': 'object',
    'properties': {
        'name': {'type': 'string'},
        'type': {'enum': ['datetime']},
        'min_val': {'type': 'string'},
        'max_val': {'type': 'string'},
        'date_format': {'type': 'string'},
        'max_clusters': {'type': 'integer'},
        'std_multiplier': {'type': 'integer'},
        'weight_threshold': {'type': 'number'}
    },
    'required': ['name', 'type'],
    'additionalProperties': False
}
_TD_ATTR_CONF = {
    'type': 'object',
    'properties': {
        'name': {'type': 'string'},
        'type': {'enum': ['timedelta']},
        'min_val': {'type': 'string'},
        'max_val': {'type': 'string'},
        'delta_format': {'type': 'string'},
        'max_clusters': {'type': 'integer'},
        'std_multiplier': {'type': 'integer'},
        'weight_threshold': {'type': 'number'}
    },
    'required': ['name', 'type'],
    'additionalProperties': False
}
_ENC_ATTR_CONF = {
    'type': 'object',
    'properties': {
        'name': {'type': 'string'},
        'type': {'enum': ['encoding']},
        'vocab_file': {'type': 'string'},
        'engine': {'enum': ['json', 'pickle', 'yaml', 'torch']}
    },
    'required': ['name', 'type', 'vocab_file'],
    'additionalProperties': False
}
_ATTR_CONF_BY_TYPE = {
    'id': _ID_ATTR_CONF,
    'categorical': _CAT_ATTR_CONF,
    'numerical': _NUM_ATTR_CONF,
    'datetime': _DT_ATTR_CONF,
    'timedelta': _TD_ATTR_CONF,
    'encoding': _ENC_ATTR_CONF
}
_ATTR_TYPE_BY_NAME: Dict[str, BaseAttribute.__class__] = {
    'id': SerialIDAttribute,
    'categorical': CategoricalAttribute,
    'numerical': NumericalAttribute,
    'datetime': DatetimeAttribute,
    'timedelta': TimedeltaAttribute,
    'encoding': EncodingAttribute
}
_DTYPE_MAP = {
    'categorical': ['categorical', 'object', 'str'],
    'numerical': ['int64', 'Int64', 'unit64', 'float64', 'Float64', 'int32', 'Int32', 'float32', 'Float32'],
    'datetime': ['datetime64', 'datetime64[ns]']
}


def create(meta: dict, values: Optional[pd.Series] = None) -> BaseAttribute:
    """
    Create attribute from meta.

    **Args**:

    - `meta` (`dict`): Metadata of the attribute described as a `dict`.
      It must have `name` and `type` key, where `type` can be 'id', 'categorical', 'numerical',
      'datetime', 'timedelta', and 'encoding'.
      It can have other keys based on the arguments to the corresponding type's attribute's constructor.
    - `values` Optional[pd.Series] (default `None`): Original value sto the attribute.

    **Return**: The created attribute.
    """
    validate(meta, _ATTR_CONF)
    validate(meta, _ATTR_CONF_BY_TYPE[meta['type']])
    kwargs = {k: v for k, v in meta.items() if k != 'type'}
    return _ATTR_TYPE_BY_NAME[meta['type']](values=values, **kwargs)


def learn_meta(data: pd.Series, is_id: bool = False, name: str = None) -> dict:
    """
    Learn an attribute's metadata `dict` object.

    **Args**:

    - `data` (`pd.Series`): The data for this attribute.
    - `is_id` (`bool`): Whether this attribute is an ID attribute. Default is `False`.
    - `name` (`str`): Name of the attribute. If not provided, the returned `dict` will not have `'name'` field.

    **Return**: A `dict` object describing the attribute's metadata.
    """
    attr_meta = {}
    if name is not None:
        attr_meta['name'] = name

    if is_id:
        attr_meta['type'] = 'id'
        return attr_meta

    for dtype, choices in _DTYPE_MAP.items():
        if str(data.dtype) in choices:
            attr_meta['type'] = dtype
            _learn_property(dtype, data, attr_meta)
            return attr_meta
    raise ValueError(f'Meta cannot be learned for {name}.')


def _learn_property(dtype: str, data: pd.Series, attr_meta: dict):
    if dtype == 'categorical':
        try:
            for d in data:
                parser.parse(str(d))
            attr_meta['type'] = 'datetime'
            return
        except (OverflowError, ValueError):
            return

    elif dtype == 'numerical':
        for i in range(-20, 20):
            rounded = data.apply(lambda x: x if pd.isnull(x) else round(x, i))
            if rounded.equals(data):
                attr_meta['rounding'] = i
                return

    elif dtype == 'datetime':
        units_to_format = {
            'y': '%Y',
            'm': '-%m',
            'd': '-%d',
            'h': ' HH',
            'min': ':mm',
            's': ':ss',
        }
        format_str = ''
        for u, format_suffix in units_to_format.items():
            format_str += format_suffix
            rounded = data.apply(lambda x: datetime.strptime(
                x.strftime(format_str), format_str))
            if rounded.equals(data):
                attr_meta['date_format'] = format_str
                return
        attr_meta['date_format'] = format_str + '.%f'
