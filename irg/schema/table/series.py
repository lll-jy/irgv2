"""Series tabular table data structure that holds data and metadata of tables in a database."""

from typing import Collection, Optional

from .table import Table


class SeriesTable(Table):
    def __init__(self, name: str, series_id: str, base_cols: Optional[Collection[str]] = None, **kwargs):
        if 'id_cols' not in kwargs:
            kwargs['id_cols'] = {}
        kwargs['id_cols'] = {*kwargs['id_cols']} | {series_id}
        super().__init__(name, ttype='series', **kwargs)
        self._base_cols = [] if base_cols is None else base_cols
        self._series_id = series_id

    def shallow_copy(self) -> "SeriesTable":
        copied = SeriesTable(
            name=self._name, series_id=self._series_id, base_cols=self._base_cols,
            ttype=self._ttype, need_fit=False, attributes=self._attr_meta,
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
