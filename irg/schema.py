import hashlib
import json
import os
import shutil
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from typing_extensions import Self

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from .utils import placeholder


class ForeignKey:
    def __init__(self,
                 child_table_name: str,
                 parent_table_name: str,
                 child_column_names: Union[str, Sequence[str]],
                 parent_column_names: Optional[Union[str, Sequence[str]]] = None,
                 unique: bool = False,
                 total_participate: bool = False):
        """
        Parameters
        ----------
        child_table_name: str
            The name of the child table.
        parent_table_name: str
            The name of the parent table.
        child_column_names: Union[str, Sequence[str]]
            The names of the child columns in the foreign key.
        parent_column_names: Union[str, Sequence[str]], optional
            The names of the parent columns in the foreign key. If this is a list, the order must correspond to
            `child_column_names`. If not provided, it will be the same as `child_column_names`.
        unique: bool
            Whether the foreign key should be unique (maximum degree = 1).
            Inference of this parameter from actual data is not an exposed functionality in the exposed research code.
        total_participate: bool
            Whether the foreign key should satisfy total participation constraint (minimum degree = 1).
            Inference of this parameter from actual data is not an exposed functionality in the exposed research code.
        """
        self.child_table_name = child_table_name
        self.parent_table_name = parent_table_name
        self.child_column_names = child_column_names if not isinstance(child_column_names, str) else [child_column_names]
        if parent_column_names is None:
            parent_column_names = self.child_column_names
        self.parent_column_names = parent_column_names if \
            not isinstance(parent_column_names, str) else [parent_column_names]
        self.unique = unique
        self.total_participate = total_participate

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ForeignKey):
            return False
        for k in ["child_table_name", "parent_table_name", "child_column_names"]:
            if getattr(self, k) != getattr(other, k):
                return False
        return True


class TableConfig:
    def __init__(self,
                 name: str,
                 primary_key: Optional[Union[str, Sequence[str]]] = None,
                 foreign_keys: Optional[Sequence[ForeignKey]] = None,
                 sortby: Optional[str] = None,
                 id_columns: Optional[Sequence[str]] = None,):
        """
        Parameters
        ----------
        name : str
            Table name.
        primary_key : Union[str, Sequence[str]], optional
            Primary key column(s). None if the table has no primary key.
        foreign_keys : Sequence[ForeignKey], optional
            Foreign key constraints on the table, with this table as child. None if the table has no foreign key.
        sortby : str, optional
            Column where the table should be (locally by the first foreign key) sorted by,
            if the table has some sequential nature.
            If not provided, this table is sorted based on the original order.
            The sortby column need to be NULL-free and continuous, and should exist only if a foreign key exists.
        id_columns : Union[str, Sequence[str]], optional
            Columns that are IDs. It means they aren't categorical or numeric (numeric ID is still ID, a criteria is
            not whether the values are comparable, as in, value 10 > value 3 and value 15 = value 3 * value 5).

        In practice, table config may contain other parameters like data types and hyperparameters for data encoding.
        """
        self.name = name
        self.primary_key = primary_key if not isinstance(primary_key, str) else [primary_key]
        self.foreign_keys = foreign_keys if foreign_keys is not None else []
        self.sortby = sortby
        self.id_columns = id_columns if id_columns is not None else []

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        foreign_keys = data.get("foreign_keys", [])
        foreign_keys = [ForeignKey(**x) for x in foreign_keys]
        data = data.copy()
        data["foreign_keys"] = foreign_keys
        return cls(**data)


class TableTransformer:
    def __init__(self, config: TableConfig):
        self.config = config
        self.columns = []
        self.categorical_columns = []
        self.numeric_columns = []
        self.agg_columns = None
        self.agg_transformer = StandardScaler() if self.config.foreign_keys else None
        self.cat_transformer = OrdinalEncoder()
        self.num_transformer = StandardScaler()
        self.split_dim = 0

    @placeholder
    def fit(self, table: pd.DataFrame):
        """
        Fit data encoders for the table.

        Parameters
        ----------
        table : pd.DataFrame
            The raw values of the table to be fitted.
        """
        print(
            "CTAB-GAN+ based data encoding should be implemented, with one-hot encoding for categorical and "
            "VGM-decomposition for continuous features. In the placeholder, label encoding and standard scaler are "
            "used instead. Moreover, in practice, the transformer should support more data types and advanced data "
            "cleaning methods when facing missing values, but the exposed version only applies naive implementations."
        )
        self.columns = table.columns
        numeric_columns = table.select_dtypes(include=np.number).columns
        categorical_columns = table.drop(columns=numeric_columns.tolist()).columns
        self.categorical_columns = [
            c for c in categorical_columns if c not in self.config.id_columns
        ]
        self.numeric_columns = [
            c for c in numeric_columns if c not in self.config.id_columns
        ]
        if self.config.foreign_keys:
            aggregated, table = self.aggregate(table)
            aggregated = aggregated.bfill().ffill()
            self.agg_columns = aggregated.columns
            self.agg_transformer.fit(aggregated.values)
        table = table.bfill().ffill()
        if self.categorical_columns:
            cat = self.cat_transformer.fit_transform(table[self.categorical_columns].values)
            self.split_dim = cat.shape[1]
        else:
            self.split_dim = 0
        if self.numeric_columns:
            self.num_transformer.fit(table[self.numeric_columns].values)

    @placeholder
    def aggregate(self, table: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Constructed the aggregated table for the given table.

        Parameters
        ----------
        table : pd.DataFrame
            Raw table values.

        Returns
        -------
        pd.DataFrame
            Aggregated data. Its index being the first foreign key columns' values.
        pd.DataFrame
            Auxiliary raw table values. Transforms sortby column into differences if exists.
        """
        print(
            "More complex aggregation functions for different data types can be used. In this exposed version, we only "
            "aggregate numeric columns, and functions used are mean, median and std."
        )
        if not self.config.foreign_keys:
            raise RuntimeError(f"Table {self.config.name} has no FK, so aggregate is not a valid operation.")
        groupby_columns = self.config.foreign_keys[0].child_column_names
        groupby = table.groupby(groupby_columns)
        if self.config.sortby:
            first_sortby: pd.Series = groupby[self.config.sortby].head(1)
            first_sortby.index = pd.MultiIndex.from_frame(
                table.loc[first_sortby.index, groupby_columns]
            )
            sorby_diff: pd.Series = groupby[self.config.sortby].diff()
            sorby_diff = sorby_diff.fillna(sorby_diff.mean())
            table = pd.concat([
                table.drop(columns=[self.config.sortby]),
                sorby_diff.to_frame(self.config.sortby)
            ], axis=1)[table.columns]
            groupby = table.groupby(groupby_columns)
            num_groupby = groupby[self.numeric_columns]
            out = num_groupby.aggregate(["mean", "median", "std"])
            if out.index.nlevels <= 1:
                out.index = pd.MultiIndex.from_arrays([out.index], names=[out.index.name])
            out = pd.concat([
                out, pd.concat({self.config.sortby: first_sortby.to_frame("first")}, axis=1)
            ], axis=1)
        else:
            num_groupby = groupby[self.numeric_columns]
            out = num_groupby.aggregate(["mean", "median", "std"])
            if out.index.nlevels <= 1:
                out.index = pd.MultiIndex.from_arrays([out.index], names=[out.index.name])
        out.columns = pd.Index([f"{a}${b}" for a, b in out.columns])
        return out, table

    @placeholder
    def transform(self, table: pd.DataFrame) -> Tuple[
        np.ndarray, Optional[Dict[Tuple, np.ndarray]], Optional[np.ndarray], Optional[pd.Index]
    ]:
        """
        Transform the given table. This should be applied after it is fitted.

        Parameters
        ----------
        table : pd.DataFrame
            The raw values of the table before transformation.

        Returns
        -------
        np.ndarray
            The encoded table, and the values are neural-network friendly. Row orders are preserved as per `table`.
        Dict[Tuple, pd.ndarray], optional
            For each set of values for the first foreign key, mapped the row indices corresponding to the values.
            This will be None if the table has no FK.
        np.ndarray, optional
            The encoded aggregated table. This will be None if the table has no FK.
        pd.Index, optional
            The aggregated table's corresponding first set of foreign key values, in the same order as the values.
            This will be None if the table has no FK.
        """
        print(
            "Transformation function that corresponding to the implemented .fit method is used, but the implemented "
            "version is oversimplified."
        )
        table = table.reset_index(drop=True)
        if self.config.foreign_keys:
            groups = table.groupby(self.config.foreign_keys[0].child_column_names).groups
            groups = {
                k: v.values for k, v in groups.items()
            }
            aggregated, table = self.aggregate(table)
            if aggregated.index.nlevels <= 1:
                groups = {(k,): v for k, v in groups.items()}
            agg_index = aggregated.index
            aggregated = self.agg_transformer.transform(aggregated.values)
        else:
            groups = None
            agg_index = None
            aggregated = None
        if self.categorical_columns:
            cat = self.cat_transformer.transform(table[self.categorical_columns].values)
        else:
            cat = np.zeros((table.shape[0], 0))
        if self.numeric_columns:
            num = self.num_transformer.transform(table[self.numeric_columns].values)
        else:
            num = np.zeros((table.shape[0], 0))
        transformed = np.concatenate([cat, num], axis=1)
        return transformed, groups, aggregated, agg_index

    @placeholder
    def inverse_transform(self, transformed: np.ndarray, groups: Optional[Dict[Tuple, np.ndarray]] = None,
                          aggregated: Optional[np.ndarray] = None, agg_index: Optional[pd.Index] = None
                          ) -> pd.DataFrame:
        """
        Inverse transform transformed data. This should be applied after it is fitted.

        Parameters
        ----------
        transformed : np.ndarray
            The first outcome in `.transform`.
        groups : Dict[Tuple, np.ndarray], optional
            The second outcome in `.transform`.
        aggregated : Optional[np.ndarray], optional
            The third outcome in `.transform`.
        agg_index : pd.Index, optional
            The fourth outcome in `.transform`.

        Returns
        -------
        pd.DataFrame
            Reconstructed data in raw values.
            ID columns are replaced by 0,1,2,....
        """
        if self.categorical_columns:
            cat = transformed[:, :self.split_dim]
            cat = np.clip(cat, 0, np.array([x.shape[0] for x in self.cat_transformer.categories_]) - 1)
            cat = self.cat_transformer.inverse_transform(cat)
            cat = pd.DataFrame(cat, columns=self.categorical_columns)
        else:
            cat = pd.DataFrame(index=np.arange(transformed.shape[0]), columns=[])
        if self.numeric_columns:
            num = self.num_transformer.inverse_transform(transformed[:, self.split_dim:])
            num = pd.DataFrame(num, columns=self.numeric_columns)
        else:
            num = pd.DataFrame(index=np.arange(transformed.shape[0]), columns=[])
        table = pd.concat([cat, num], axis=1)
        for c in self.config.id_columns:
            table[c] = np.arange(table.shape[0])
        table = table[self.columns]

        if self.config.foreign_keys:
            groupby_columns = self.config.foreign_keys[0].child_column_names
            for vals, idx in groups.items():
                table.loc[idx, groupby_columns] = pd.Series(
                    {c: v for c, v in zip(groupby_columns, vals)}
                ).to_frame().T.loc[[0] * idx.shape[0]].set_axis(idx, axis=0)
            if self.config.sortby:
                aggregated = self.agg_transformer.inverse_transform(aggregated)
                aggregated = pd.DataFrame(aggregated, index=agg_index, columns=self.agg_columns)
                first_sortby = aggregated[f"{self.config.sortby}$first"]
                head = table.groupby(groupby_columns)[groupby_columns].head(1)
                agg_idx_to_table_idx = {
                    tuple(row[groupby_columns]): i for i, row in head.iterrows()
                }
                first_sortby.index = [agg_idx_to_table_idx[x] for x in first_sortby.index]
                table.loc[head.index, self.config.sortby] = first_sortby
                table[self.config.sortby] = table.groupby(groupby_columns)[self.config.sortby].cumsum()
        return table

    @classmethod
    def load(cls, path: str) -> Self:
        """Load .pt file from path."""
        return torch.load(path)

    def save(self, path: str):
        """Save .pt file to path."""
        torch.save(self, path)


class RelationalTransformer:
    def __init__(self,
                 tables: Dict[str, TableConfig],
                 order: List[str],
                 max_ctx_dim: int = 500):
        """
        Parameters
        ----------
        tables : Dict[str, TableConfig]
            All tables and their properties.
            Inference and validation of this parameter from actual data is not an exposed functionality in the
            exposed research code.
        order : List[str]
            The order of the tables to be generated. User must provide the order following a valid topological order.
        max_ctx_dim : int=500
            The maximum context dimension, otherwise its dimension will be reduced.
        """
        self.order = order
        self.transformers = {}
        self.children: Dict[str, List[ForeignKey]] = defaultdict(list)
        for tn in order:
            config = tables[tn]
            self.transformers[tn] = TableTransformer(config)
            for fk in config.foreign_keys:
                self.children[fk.parent_table_name].append(fk)
        self.max_ctx_dim = max_ctx_dim
        self._fitted_cache_dir = None
        self._sizes_of = {}
        self._nullable = {}
        self._parent_dims = {}
        self._core_dims = {}

    def fit(self, tables: Dict[str, str], cache_dir: str = "./cache"):
        """
        Fit the data encoding and processor, and transformed data will be output to cache_dir too.

        Parameters
        ----------
        tables : Dict[str, str]
            Tables' names mapped to their paths (csv files).
            Please reserve column names starting with "_" and consisting of "$" for internal processing purpose.
            Although missing values will not fail this version of code, no N/A will be generated except for N/As in
            foreign keys, by the naiveness of this version of data transformer.
        cache_dir : str
            The directory for caching the output files. It contains:

            - `TABLE_NAME.csv`: Raw table values.
            - `TABLE_NAME-transformer.pt`: saved transformer.
            - `TABLE_NAME.pt`: Transformed data from the raw values.
        """
        self._fitted_cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        for tn in self.order:
            table = pd.read_csv(tables[tn])
            self._sizes_of[tn] = table.shape[0]
            table.to_csv(os.path.join(cache_dir, f"{tn}.csv"), index=False)
            transformer = self.transformers[tn]
            transformer.fit(table)
            transformer.save(os.path.join(cache_dir, f"{tn}-transformer.pt"))

            foreign_keys = self.transformers[tn].config.foreign_keys
            if foreign_keys:
                self._nullable[tn] = []
                encoded, groups, aggregated, agg_index = transformer.transform(table)
                torch.save({
                    "actual": (None, None, encoded, None)
                }, os.path.join(cache_dir, f"{tn}.pt"))
                key, context, new_encoded = self._extend_till(tn, tn, table.columns.tolist(), cache_dir)
                if not (encoded == new_encoded[:, self._core_dims[tn]]).all() or not key.equals(table):
                    raise RuntimeError("Error when extending.")

                agg_context = np.zeros((aggregated.shape[0], 0))
                actual_context = np.zeros((aggregated.shape[0], 0))
                transformed_context = np.zeros((encoded.shape[0], 0))
                length = np.zeros(aggregated.shape[0])
                all_fk_info = []
                for fi, fk in enumerate(foreign_keys):
                    fk_info = {}
                    parent_key, parent_context, parent_encoded = self._extend_till(
                        fk.parent_table_name, tn, fk.parent_column_names, cache_dir, fitting=False, queue=[fk]
                    )
                    degree_x = np.concatenate([parent_context, parent_encoded], axis=1)
                    degree_y = table[fk.child_column_names].groupby(fk.child_column_names).size()
                    parent_key_as_child = parent_key.rename(columns={
                        p: c for p, c in zip(fk.parent_column_names, fk.child_column_names)
                    })
                    y_order = pd.MultiIndex.from_frame(parent_key_as_child)
                    if degree_y.index.nlevels <= 1:
                        degree_y.index = pd.MultiIndex.from_arrays([degree_y.index], names=[degree_y.index.name])
                    degree_y = degree_y.loc[y_order]
                    degree_y = degree_y.values
                    if fi == 0:
                        non_zero_degree_x = pd.DataFrame(
                            degree_x, columns=[f"_dim{i:02d}" for i in range(degree_x.shape[-1])],
                            index=parent_key.index
                        )
                        non_zero_degree_x = pd.concat([parent_key_as_child, non_zero_degree_x], axis=1)
                        agg_context = agg_index.to_frame().reset_index(drop=True)
                        agg_context = agg_context.merge(
                            non_zero_degree_x, how="left", on=agg_index.names
                        )
                        agg_context = agg_context.set_index(agg_index.names)
                        if agg_context.index.nlevels <= 1:
                            agg_context.index = pd.MultiIndex.from_arrays([agg_context.index], names=agg_index.names)
                        agg_context = agg_context.loc[agg_index].values
                        length = degree_y

                        actual_context = np.concatenate([agg_context, aggregated], axis=1)
                        transformed_context = np.empty((encoded.shape[0], actual_context.shape[-1]))
                        for g, idx in groups.items():
                            transformed_context[idx] = actual_context[g]
                    fk_info["degree"] = degree_x, degree_y

                    if table[fk.child_column_names].isna().any().any():
                        isna_y = table[fk.child_column_names].isna().any(axis=1)
                        fk_info["isna"] = np.concatenate([transformed_context, new_encoded], axis=1), isna_y.values
                        self._nullable[tn].append(True)
                    else:
                        self._nullable[tn].append(False)
                    all_fk_info.append(fk_info)

                out = {
                    "aggregated": (agg_context, aggregated),
                    "actual": (
                        actual_context, length, new_encoded,
                        [groups[tuple(x) if isinstance(x, tuple) else (x,)] for x in agg_index]
                    ),
                    "foreign_keys": all_fk_info,
                }
            else:
                encoded, _, _, _ = transformer.transform(table)
                out = {
                    "encoded": encoded
                }
            torch.save(out, os.path.join(cache_dir, f"{tn}.pt"))

    @placeholder
    def _extend_till(self, table: str, till: str, keys: Sequence[str], cache_dir: str,
                     fitting: bool = True, queue: List[ForeignKey] = []) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        print("N/As created by left outer join are actually processed by smarter ways.")
        allowed_tables = self.order[:self.order.index(till)]
        raw = pd.read_csv(os.path.join(cache_dir, f"{table}.csv"))
        if self.transformers[table].config.foreign_keys:
            _, _, encoded, _ = self.actual_generation_for(table, cache_dir)
        else:
            encoded = self.standalone_encoded_for(table, cache_dir)
        core_columns = [f"_dim{i:02d}" for i in range(encoded.shape[-1])]
        core = pd.DataFrame(encoded, columns=core_columns, index=raw.index)
        core = pd.concat([raw.index.to_frame(False, "_id"), raw, core], axis=1)
        for fi, fk in enumerate(self.transformers[table].config.foreign_keys):
            if fk in queue:
                continue
            parent_raw, parent_context, parent_encoded = self._extend_till(
                fk.parent_table_name, till, fk.parent_column_names, cache_dir, fitting, queue + [fk]
            )
            parent_encoded = np.concatenate([parent_context, parent_encoded], axis=1)
            parent_encoded = pd.DataFrame(
                parent_encoded, columns=[f"_dim{i:02d}" for i in range(parent_encoded.shape[-1])],
                index=np.arange(parent_encoded.shape[0])
            )
            parent_idx_df = parent_raw[fk.parent_column_names].rename(columns={
                p: c for p, c in zip(fk.parent_column_names, fk.child_column_names)
            })
            parent_encoded = pd.concat([parent_idx_df, parent_encoded], axis=1)
            core = core.merge(parent_encoded, on=fk.child_column_names, how="left", suffixes=("", f"_p{fi}")).fillna(0)

        for fi, fk in enumerate(self.children[table]):
            if fk.child_table_name not in allowed_tables or fk in queue:
                continue
            sibling_raw, sibling_context, sibling_encoded = self._extend_till(
                fk.child_table_name, till, fk.child_column_names, cache_dir, fitting, queue + [fk]
            )
            sibling_encoded = np.concatenate([sibling_context, sibling_encoded], axis=1)
            sibling_encoded = self._reduce_dims(sibling_encoded, table, fitting, queue, cache_dir, allowed_tables)
            encoded_columns = [f"_dim{i:02d}" for i in range(sibling_encoded.shape[-1])]
            sibling_encoded = pd.DataFrame(
                sibling_encoded, columns=encoded_columns, index=np.arange(sibling_encoded.shape[0])
            )
            sibling_idx_df = sibling_raw[fk.child_column_names].rename(columns={
                c: p for c, p in zip(fk.child_column_names, fk.parent_column_names)
            })
            sibling_encoded = pd.concat([sibling_idx_df, sibling_encoded], axis=1)
            sibling_encoded_aggregated = sibling_encoded.groupby(fk.parent_column_names).aggregate(["mean", "std"])
            sibling_encoded_aggregated = sibling_encoded_aggregated.reset_index()
            sibling_encoded_aggregated.columns = pd.Index([
                f"{a}${b}" if b else a for a, b in sibling_encoded_aggregated.columns
            ])
            core = core.merge(
                sibling_encoded_aggregated, on=fk.parent_column_names, how="left", suffixes=("", f"_c{fi}")
            ).fillna(0)

        core = core.set_index("_id").loc[raw.index]
        raw_keys = raw[keys]
        context_columns = [c for c in core.columns if c.startswith("_dim") and c.endswith("_p0")]
        context = core[context_columns]
        encoded = core.drop(columns=context_columns + raw.columns.tolist())

        if fitting and table == till:
            parent_dims = [None]
            name_to_id = {
                c: i for i, c in enumerate(encoded.columns)
            }
            for fi in range(1, len(self.transformers[table].config.foreign_keys)):
                parent_dims.append([
                    name_to_id[n] for n in encoded.columns if n.endswith(f"_p{fi}") and n.startswith("_dim")
                ])
            self._parent_dims[table] = parent_dims
            self._core_dims[table] = [name_to_id[n] for n in core_columns]
        return raw_keys, context.values, encoded.values

    def _reduce_dims(self, parent_encoded: np.ndarray, table: str, fitting: bool, queue: List[ForeignKey],
                     cache_dir: str, allowed_tables: List[str]) -> np.ndarray:
        if parent_encoded.shape[-1] > self.max_ctx_dim:
            queue_str = json.dumps([
                f"parent={qfk.parent_table_name}, child={qfk.child_table_name}, "
                f"columns={qfk.child_column_names}" for qfk in queue
            ])
            pca_name = f"{table}_{len(allowed_tables)}_{hashlib.sha1(queue_str.encode()).hexdigest()}"
            os.makedirs(os.path.join(cache_dir, "pca"), exist_ok=True)
            pca_path = os.path.join(cache_dir, "pca", f"{pca_name}.pt")
            if fitting:
                if os.path.exists(pca_path):
                    raise FileExistsError("File for PCA already exists.")
                pca = PCA(n_components=self.max_ctx_dim)
                parent_encoded = pca.fit_transform(parent_encoded)
                torch.save(pca, pca_path)
            else:
                pca = torch.load(pca_path)
                parent_encoded = pca.transform(parent_encoded)
        return parent_encoded

    def fitted_size_of(self, table_name: str) -> int:
        """
        Table size for fitting.

        Parameters
        ----------
        table_name : str
            The table name to get size of.

        Returns
        -------
        int
            Number of rows in the table.
        """
        return self._sizes_of[table_name]

    @classmethod
    def standalone_encoded_for(cls, table_name: str, cache_dir: str = "./cache") -> np.ndarray:
        """
        Get the standalone single table (table without parent) encoded values for neural network generators.

        Parameters
        ----------
        table_name : str
            The table name to extract data from.
        cache_dir : str
            The directory for cached files. It should be the same as `.fit`.

        Returns
        -------
        np.ndarray
            The encoded data for neural network generators.
        """
        return torch.load(os.path.join(cache_dir, f"{table_name}.pt"))["encoded"]

    @classmethod
    def degree_prediction_for(cls, table_name: str, fk_idx: int, cache_dir: str = "./cache") -> Tuple[
        np.ndarray, Optional[np.ndarray]
    ]:
        """
        Get the X and y for degree prediction.

        Parameters
        ----------
        table_name : str
            The table name to extract data from.
        fk_idx : int
            The foreign key index on the table to extract data from.
        cache_dir : str
            The directory for cached files. It should be the same as `.fit` or `sampled_dir` for `.prepare_sampled_dir`.

        Returns
        -------
        np.ndarray
            The X for degree prediction. Rows are in correspondence with the parent table.
        np.ndarray, optional
            The y for degree prediction. Rows are in correspondence with the parent table.
            During generation, this will be None.
        """
        return torch.load(os.path.join(cache_dir, f"{table_name}.pt"))["foreign_keys"][fk_idx]["degree"]

    @classmethod
    def isna_indicator_prediction_for(cls, table_name: str, fk_idx: int, cache_dir: str = "./cache") -> Optional[Tuple[
        np.ndarray, Optional[np.ndarray]
    ]]:
        """
        Get the X and y for is-N/A indicator prediction.

        Parameters
        ----------
        table_name : str
            The table name to extract data from.
        fk_idx : int
            The foreign key index on the table to extract data from.
        cache_dir : str
            The directory for cached files. It should be the same as `.fit` or `sampled_dir` for `.prepare_sampled_dir`.

        Returns
        -------
        np.ndarray
            The X for is-N/A prediction. Rows are in correspondence with the child table.
            Returns None without a tuple if this foreign key is not nullable.
        np.ndarray, optional
            The y for is-N/A prediction. Rows are in correspondence with the child table.
            During generation, this will be None.
        """
        return torch.load(os.path.join(cache_dir, f"{table_name}.pt"))["foreign_keys"][fk_idx].get("isna")

    @classmethod
    def aggregated_generation_for(cls, table_name: str, cache_dir: str = "./cache") -> Tuple[
        np.ndarray, Optional[np.ndarray]
    ]:
        """
        Data for aggregated information generation.

        Parameters
        ----------
        table_name : str
            The table name to extract data from.
        cache_dir : str
            The directory for cached files. It should be the same as `.fit` or `sampled_dir` for `.prepare_sampled_dir`.

        Returns
        -------
        np.ndarray
            The context for aggregated information generation.
        np.ndarray, optional
            The aggregated information to be generated. During generation, this will be None.
        """
        return torch.load(os.path.join(cache_dir, f"{table_name}.pt"))["aggregated"]

    @classmethod
    def actual_generation_for(cls, table_name: str, cache_dir: str = "./cache") -> Tuple[
        np.ndarray, np.ndarray, Optional[np.ndarray], Optional[List[np.ndarray]]
    ]:
        """
        Data for actual data generation.

        Parameters
        ----------
        table_name : str
            The table name to extract data from.
        cache_dir : str
            The directory for cached files. It should be the same as `.fit` or `sampled_dir` for `.prepare_sampled_dir`.

        Returns
        -------
        np.ndarray
            The static context for actual information generation. The length and order should follow the first parent
            of the table.
        np.ndarray
            The lengths for aggregated information generation. The length and order should follow the first parent
            of the table.
        np.ndarray, optional
            The actual data to be generated. During generation, this will be None. The length and order should follow
            this table.
        List[np.ndarray], optional
            The indices in actual data corresponding to each parent row to be generated. During generation, this
            will be None. The length and order should follow the first parent of the table, and the content for
            row IDs should follow the row indices in the actual data to be generated.
        """
        return torch.load(os.path.join(cache_dir, f"{table_name}.pt"))["actual"]

    @placeholder
    def fk_matching_for(self, table_name: str, fk_idx: int, sampled_dir: str = "./cache") -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Optional[np.ndarray]], List[np.ndarray]
    ]:
        """
        Data for matching. This step applies to the generated data only.

        Parameters
        ----------
        table_name : str
            The table name to extract data from.
        fk_idx : int
            The foreign key index on the table to extract data from.
        sampled_dir : str
            The directory for sampled files. It should be the same as `.prepare_sampled_dir`.

        Returns
        -------
        np.ndarray
            The current table's dimensions corresponding to this foreign key's parent.
        np.ndarray
            The parent table's encoded data, with the same number of dimensions as the first item in the tuple.
        np.ndarray
            The degrees for the parent table, corresponding to each row in the second item in the tuple.
        np.ndarray
            The is-N/A indicator for this foreign key, corresponding to each row in the first item in the tuple.
        List[Optional[np.ndarray]]
            For each row in the real data, the allowed set of indices to be matched in the parent table, so the list
            index are row indices in the current table, and values in the list are row indices in the parent table.
            This value is supposed to handle overlapping foreign keys. If it is None, it means no constraint is
            placed on it.
        List[np.ndarray]
            The set of row indices in the current table that are not supposed to match to the same row in the parent.
            The values are row indices in the current table. This value is supposed to handle composite primary keys
            by multiple foreign keys.
        """
        print("In actual IRG, the fifth and sixth arguments are calculated, but we hide the implementation from "
              "the public version. Thus, overlapping key constraints are invalid input for this simplified version. "
              "However, placeholder output is returned.")
        loaded = torch.load(os.path.join(sampled_dir, f"{table_name}.pt"))
        _, _, values, _ = loaded["actual"]
        values = values[:, self._parent_dims[table_name][fk_idx]]
        parent, degrees = loaded["foreign_keys"][fk_idx]["degree"]
        if values.shape[-1] != parent.shape[-1]:
            raise RuntimeError(f"The sizes to be matched are different: {values.shape}, {parent.shape}.")
        isnull = loaded["foreign_keys"][fk_idx]["isna"]
        if isnull is None:
            isna = np.zeros(values.shape[0], dtype=np.bool_)
        else:
            _, isna = isnull
        return values, parent, degrees, isna, [None] * values.shape[0], []
    
    def prepare_sampled_dir(self, sampled_dir: str):
        """
        Prepare directory for sampled data.
        
        Parameters
        ----------
        sampled_dir : str
            The directory for sampled data.
        """
        if os.path.exists(sampled_dir):
            shutil.rmtree(sampled_dir)
        os.makedirs(sampled_dir, exist_ok=True)
        if os.path.exists(os.path.join(self._fitted_cache_dir, "pca")):
            shutil.copytree(os.path.join(self._fitted_cache_dir, "pca"), os.path.join(sampled_dir, "pca"))

    @classmethod
    def save_standalone_encoded_for(cls, table_name: str, encoded: np.ndarray, sampled_dir: str = "./sampled"):
        """
        After the standalone encoded data is generated, save the encoded data to disk.

        Parameters
        ----------
        table_name : str
            The table name to save.
        encoded : np.ndarray
            The encoded data to be saved.
        sampled_dir : str
            The directory for sampled files. It should be the same as `.prepare_sampled_dir`.
        """
        torch.save({"encoded": encoded}, os.path.join(sampled_dir, f"{table_name}.pt"))

    @classmethod
    def save_degree_for(cls, table_name: str, fk_idx: int, degree: np.ndarray, sampled_dir: str = "./sampled"):
        """
        After the degree data is generated, save the degree data to disk.

        Parameters
        ----------
        table_name : str
            The table name to save.
        fk_idx : int
            The index of foreign key to save for this degree.
        degree : np.ndarray
            The degree values to save.
        sampled_dir : str
            The directory for sampled files. It should be the same as `.prepare_sampled_dir`.
        """
        loaded = torch.load(os.path.join(sampled_dir, f"{table_name}.pt"))
        x, _ = loaded["foreign_keys"][fk_idx]["degree"]
        loaded["foreign_keys"][fk_idx]["degree"] = x, degree

        if fk_idx == 0:
            a, b, c, d = loaded.get("actual", (None, None, None, None))
            non_zero_deg = degree > 0
            loaded["actual"] = a, degree[non_zero_deg], c, d
            non_zero_x = x[non_zero_deg]
            loaded["aggregated"] = non_zero_x, None

        torch.save(loaded, os.path.join(sampled_dir, f"{table_name}.pt"))

    def save_isna_indicator_for(self, table_name: str, fk_idx: int, isna: np.ndarray, sampled_dir: str = "./sampled"):
        """
        After the is-N/A indicator data is generated, save the is-N/A indicator data to disk.

        Parameters
        ----------
        table_name : str
            The table name to save.
        fk_idx : int
            THe index of foreign key to save for this is-N/A indicator.
        isna : np.ndarray
            The is-N/A indicator data to be saved.
        sampled_dir : str
            The directory for sampled files. It should be the same as `.prepare_sampled_dir`.
        """
        loaded = torch.load(os.path.join(sampled_dir, f"{table_name}.pt"))
        x, _ = loaded["foreign_keys"][fk_idx]["isna"]
        loaded["foreign_keys"][fk_idx]["isna"] = x, isna
        a, b, encoded, d = loaded["actual"]
        encoded[:, self._parent_dims[table_name][fk_idx]] = 0
        loaded["actual"] = a, b, encoded, d

        torch.save(loaded, os.path.join(sampled_dir, f"{table_name}.pt"))

    @classmethod
    def save_aggregated_info_for(cls, table_name: str, aggregated: np.ndarray, sampled_dir: str = "./sampled"):
        """
        After the aggregated information data is generated, save the aggregated information to disk.

        Parameters
        ----------
        table_name : str
            The table name to save.
        aggregated : np.ndarray
            The aggregated information values to save.
        sampled_dir : str
            The directory for sampled files. It should be the same as `.prepare_sampled_dir`.
        """
        loaded = torch.load(os.path.join(sampled_dir, f"{table_name}.pt"))
        agg_context, _ = loaded["aggregated"]
        loaded["aggregated"] = agg_context, aggregated
        actual_context = np.concatenate([agg_context, aggregated], axis=1)
        _, length, _, _ = loaded["actual"]
        loaded["actual"] = actual_context, length, None, None

        torch.save(loaded, os.path.join(sampled_dir, f"{table_name}.pt"))

    @classmethod
    def save_actual_values_for(
            cls, table_name: str, values: np.ndarray, groups: List[np.ndarray], sampled_dir: str = "./sampled"
    ):
        """
        After the actual values data is generated, save the actual values data to disk.

        Parameters
        ----------
        table_name : str
            The table name to save.
        values : np.ndarray
            The actual values generated to save.
        groups : List[np.ndarray]
            The grouping information to save.
        sampled_dir : str
            The directory for sampled files. It should be the same as `.prepare_sampled_dir`.
        """
        loaded = torch.load(os.path.join(sampled_dir, f"{table_name}.pt"))
        context, length, _, _ = loaded["actual"]
        loaded["actual"] = context, length, values, groups
        for i, fk in enumerate(loaded["foreign_keys"]):
            isnull = fk["isna"]
            if isnull is not None:
                cids = np.repeat(np.arange(context.shape[0]), length.astype(int))
                loaded["foreign_keys"][i]["isna"] = np.concatenate([context[cids], values], axis=1), None
                break
        torch.save(loaded, os.path.join(sampled_dir, f"{table_name}.pt"))

    def save_matched_indices_for(self, table_name: str, fk_idx: int,
                                 indices: np.ndarray, sampled_dir: str = "./sampled"):
        """
        After foreign key matching, save the matched indices to disk.

        Parameters
        ----------
        table_name : str
            The table name to save.
        fk_idx : int
            The index of foreign key to save for this indices.
        indices : np.ndarray
            The matched indices to save.
        sampled_dir : str
            The directory for sampled files. It should be the same as `.prepare_sampled_dir`.
        """
        loaded = torch.load(os.path.join(sampled_dir, f"{table_name}.pt"))
        loaded["foreign_keys"][fk_idx]["match"] = indices
        context, length, encoded, d = loaded["actual"]
        parent, _ = loaded["foreign_keys"][fk_idx]["degree"]
        isna = np.isnan(indices)

        encoded[np.ix_(~isna, self._parent_dims[table_name][fk_idx])] = parent[indices[~isna]]
        loaded["actual"] = context, length, encoded, d
        for i, fk in enumerate(loaded["foreign_keys"]):
            if i <= fk_idx:
                continue
            isnull = fk["isna"]
            if isnull is not None:
                cids = np.repeat(np.arange(context.shape[0]), length.astype(int))
                loaded["foreign_keys"][i]["isna"] = np.concatenate([context[cids], encoded], axis=1), None
                break
        torch.save(loaded, os.path.join(sampled_dir, f"{table_name}.pt"))

    def copy_fitted_for(self, table_name: str, sampled_dir: str = "./sampled"):
        """
        Copy real fitting data if the table need not be synthesized.

        Parameters
        ----------
        table_name : str
            The name of the table.
        sampled_dir : str
            The directory for sampled results.
        """
        shutil.copyfile(os.path.join(self._fitted_cache_dir, f"{table_name}.pt"),
                        os.path.join(sampled_dir, f"{table_name}.pt"))
        shutil.copyfile(os.path.join(self._fitted_cache_dir, f"{table_name}.csv"),
                        os.path.join(sampled_dir, f"{table_name}.csv"))

    def prepare_next_for(self, table_name: str, sampled_dir: str = "./cache"):
        """
        Prepare next table to be generated.

        Parameters
        ----------
        table_name : str
            The table just finished.
        sampled_dir : str
            The directory where sampled data are stored.
        """
        if self.transformers[table_name].config.foreign_keys:
            _, aggregated = self.aggregated_generation_for(table_name, sampled_dir)
            _, _, encoded, indices = self.actual_generation_for(table_name, sampled_dir)
            foreign_keys = self.transformers[table_name].config.foreign_keys
            fk = foreign_keys[0]
            parent = pd.read_csv(os.path.join(sampled_dir, f"{fk.parent_table_name}.csv"))
            parent_idx = pd.MultiIndex.from_frame(parent[fk.parent_column_names].rename({
                p: c for p, c in zip(fk.parent_column_names, fk.child_column_names)
            }))
            groups = {
                pi: idx for pi, idx in zip(parent_idx, indices)
            }
            recovered = self.transformers[table_name].inverse_transform(
                encoded[:, self._core_dims[table_name]], groups, aggregated, parent_idx
            )

            occurred_cols = set()
            for i, fk in enumerate(foreign_keys):
                if i == 0:
                    occurred_cols |= set(fk.child_column_names)
                    continue
                isnull = self.isna_indicator_prediction_for(table_name, i, sampled_dir)
                if isnull is not None:
                    _, isna = isnull
                    recovered.loc[isna, [c for c in fk.child_column_names if c not in occurred_cols]] = np.nan
                occurred_cols |= set(fk.child_column_names)
        else:
            encoded = self.standalone_encoded_for(table_name, sampled_dir)
            recovered = self.transformers[table_name].inverse_transform(encoded)
        recovered.to_csv(os.path.join(sampled_dir, f"{table_name}.csv"), index=False)

        table_idx = self.order.index(table_name)
        if table_idx >= len(self.order) - 1:
            return
        next_table_name = self.order[table_idx + 1]
        degrees = []
        for i, fk in enumerate(self.transformers[next_table_name].config.foreign_keys):
            parent_raw, parent_context, parent_encoded = self._extend_till(
                fk.parent_table_name, next_table_name, fk.parent_column_names, sampled_dir, False, [fk]
            )
            parent_extend_till = np.concatenate([parent_context, parent_encoded], axis=1)
            degrees.append(parent_extend_till)
        torch.save({
            "foreign_keys": [{
                "degree": (x, None), "isna": (None, None) if y else None
            } for x, y in zip(degrees, self._nullable.get(next_table_name, []))]
        }, os.path.join(sampled_dir, f"{next_table_name}.pt"))
