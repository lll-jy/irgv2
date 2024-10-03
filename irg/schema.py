import hashlib
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Self, Sequence, Tuple, Union

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
                 id_columns: Optional[Union[str, Sequence[str]]] = None,):
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
            out = pd.concat([
                out, pd.concat({self.config.sortby: first_sortby.to_frame("first")}, axis=1)
            ], axis=1)
        else:
            num_groupby = groupby[self.numeric_columns]
            out = num_groupby.aggregate(["mean", "median", "std"])
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
            cat = self.cat_transformer.inverse_transform(transformed[:, :self.split_dim])
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
                table.loc[idx, groupby_columns] = pd.Series({c: v for c, v in zip(groupby_columns, vals)})
            if self.config.sortby:
                aggregated = self.agg_transformer.inverse_transform(aggregated)
                aggregated = pd.DataFrame(aggregated, index=agg_index, columns=self.agg_columns)
                first_sortby = aggregated[self.config.sortby]["first"]
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

    def fit(self, tables: Dict[str, str], cache_dir: str = "./cache"):
        """
        Fit the data encoding and processor, and transformed data will be output to cache_dir too.

        Parameters
        ----------
        tables : Dict[str, str]
            Tables' names mapped to their paths (csv files).
            Please reserve column names starting with "_" and consisting of "$" for internal processing purpose.
        cache_dir : str
            The directory for caching the output files. It contains:

            - `TABLE_NAME.csv`: Raw table values.
            - `TABLE_NAME-transformer.pt`: saved transformer.
            - `TABLE_NAME.pt`: Transformed data from the raw values.
        """
        os.makedirs(cache_dir, exist_ok=True)
        for tn in self.order:
            table = pd.read_csv(tables[tn])
            table.to_csv(os.path.join(cache_dir, f"{tn}.csv"), index=False)
            transformer = self.transformers[tn]
            transformer.fit(table)
            transformer.save(os.path.join(cache_dir, f"{tn}-transformer.pt"))

            foreign_keys = self.transformers[tn].config.foreign_keys
            if foreign_keys:
                encoded, groups, aggregated, agg_index = transformer.transform(table)
                torch.save({
                    "actual": (None, None, encoded, None)
                }, os.path.join(cache_dir, f"{tn}.pt"))
                key, context, new_encoded = self._extend_till(tn, tn, table.columns.tolist(), cache_dir)
                if not (encoded == new_encoded).all() or not key.equals(table):
                    raise RuntimeError("Error when extending.")

                agg_context = np.zeros((aggregated.shape[0], 0))
                length = np.zeros(aggregated.shape[0])
                all_fk_info = []
                for fi, fk in enumerate(foreign_keys):
                    fk_info = {}
                    parent_key, parent_context, parent_encoded = self._extend_till(
                        fk.parent_table_name, tn, fk.parent_column_names, cache_dir, False
                    )
                    degree_x = np.concatenate([parent_context, parent_encoded], axis=1)
                    degree_y = table[fk.child_column_names].groupby(fk.child_column_names).size()
                    y_order = pd.MultiIndex.from_frame(parent_key.rename(columns={
                        p: c for p, c in zip(fk.parent_column_names, fk.child_column_names)
                    }))
                    degree_y = degree_y.loc[y_order]
                    degree_y = degree_y.values
                    if fi == 0:
                        agg_context = degree_x
                        length = degree_y
                    fk_info["degree"] = degree_x, degree_y

                    if table[fk.child_column_names].isna().any().any():
                        isna_y = table[fk.child_column_names].isna().any(axis=1)
                        fk_info["isna"] = np.concatenate([context, encoded], axis=1), isna_y.values
                    all_fk_info.append(fk_info)

                out = {
                    "aggregated": (agg_context, aggregated),
                    "actual": (
                        np.concatenate([agg_context, aggregated], axis=1), length, encoded,
                        [groups[tuple(x)] for x in agg_index]
                    ),
                    "foreign_keys": all_fk_info,
                }
            else:
                encoded, _, _, _ = transformer.transform(table)
                out = {
                    "encoded": encoded
                }
            torch.save(out, os.path.join(cache_dir, f"{tn}.pt"))

    def _extend_till(self, table: str, till: str, keys: Sequence[str], cache_dir: str,
                     fitting: bool = True, queue: List[ForeignKey] = []) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        allowed_tables = self.order[:self.order.index(till)]
        raw = pd.read_csv(os.path.join(cache_dir, f"{table}.csv"))
        _, _, encoded, _ = self.actual_generation_for(table, cache_dir)
        core_columns = [f"_dim{i:02d}" for i in range(encoded.shape[-1])]
        core = pd.DataFrame(encoded, columns=core_columns, index=raw.index)
        core = pd.concat([raw, core], axis=1)
        for fi, fk in enumerate(self.transformers[table].config.foreign_keys):
            if fk in queue:
                continue
            parent_raw, parent_context, parent_encoded = self._extend_till(
                fk.parent_table_name, till, fk.parent_column_names, cache_dir, fitting, queue + [fk]
            )
            parent_encoded = np.concatenate([parent_context, parent_encoded], axis=1)
            parent_encoded = self._reduce_dims(parent_encoded, table, fitting, queue, cache_dir, allowed_tables)
            parent_encoded = pd.DataFrame(
                parent_encoded, columns=[f"_dim{i:02d}" for i in range(parent_encoded.shape[-1])],
                index=np.arange(parent_encoded.shape[0])
            )
            parent_idx_df = parent_raw[fk.parent_column_names].rename(columns={
                p: c for p, c in zip(fk.parent_column_names, fk.child_column_names)
            })
            parent_encoded = pd.concat([parent_idx_df, parent_encoded], axis=1)
            core = core.merge(parent_encoded, on=fk.child_column_names, how="left", suffixes=("", f"_p{fi}"))

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
            )

        raw_keys = core[keys]
        context = core.drop(columns=raw.columns.tolist() + core_columns).values
        encoded = core[core_columns].values
        return raw_keys, context, encoded

    def _reduce_dims(self, parent_encoded: np.ndarray, table: str, fitting: bool, queue: List[ForeignKey],
                     cache_dir: str, allowed_tables: List[str]) -> np.ndarray:
        if parent_encoded.shape[-1] > self.max_ctx_dim:
            queue_str = json.dumps([
                f"parent={qfk.parent_table_name}, child={qfk.child_table_name}, "
                f"columns={qfk.child_column_names}" for qfk in queue
            ])
            pca_name = f"{table}_{len(allowed_tables)}_{hashlib.sha1(queue_str.encode()).hexdigest()}"
            pca_path = os.path.join(cache_dir, f"{pca_name}.pt")
            if os.path.exists(pca_path):
                raise FileExistsError("File for PCA already exists.")
            if fitting:
                pca = PCA(n_components=self.max_ctx_dim)
                parent_encoded = pca.fit_transform(parent_encoded)
                torch.save(pca, pca_path)
            else:
                pca = torch.load(pca_path)
                parent_encoded = pca.transform(parent_encoded)
        return parent_encoded

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
            The directory for cached files. It should be the same as `.fit` or `.generate`.

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
            The directory for cached files. It should be the same as `.fit` or `.generate`.

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
            The directory for cached files. It should be the same as `.fit` or `.generate`.

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
            The directory for cached files. It should be the same as `.fit` or `.generate`.

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
