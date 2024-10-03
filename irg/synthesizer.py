import os
from typing import Any, Dict, List

import numpy as np

from .schema import RelationalTransformer, TableConfig
from .standalone import train_standalone, generate_standalone
from .degree import train_degrees, predict_degrees


class IncrementalRelationalGenerator:
    def __init__(self, tables: Dict[str, TableConfig],
                 order: List[str],
                 max_ctx_dim: int = 500,
                 default_args: Dict[str, Any] = {}, table_specific_args: Dict[str, Dict[str, Any]] = {}):
        """
        Parameters
        ----------
        tables, order, max_ctx_dim
            Arguments to `RelationalTransformer`.
        default_args : Dict[str, Any]
            Default arguments for models. One can refer to the sample config file in `config/football.yaml` key
            `default_args` as an example, and we also added description for each value there.
        table_specific_args : Dict[str, Dict[str, Any]]
            Table-specific arguments for models. Keys are table names and values have the same structure as
            `default_args`.
        """
        self.transformer = RelationalTransformer(tables, order, max_ctx_dim)
        self.model_args = {}
        for t in self.transformer.order:
            if self.transformer.transformers[t].config.foreign_keys:
                keys = ["degree", "isna",]
            else:
                keys = ["standalone"]
            all_keys = set(table_specific_args.get(t, {}).keys()) | set(default_args.keys())
            out_args = {}
            for k in all_keys:
                if k in keys:
                    out_args[k] = table_specific_args.get(t, {}).get(k, {}) | default_args.get(k, {})
                else:
                    value = table_specific_args.get(t, {}).get(k, default_args.get(k, {}))
                    if not isinstance(value, Dict):
                        out_args[k] = value
            for k in keys:
                if k not in out_args:
                    out_args[k] = {}
            self.model_args[t] = out_args

    def fit(self, tables: Dict[str, str], out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        data_cache_dir = os.path.join(out_dir, "data")
        model_cache_dir = os.path.join(out_dir, "model")
        self.transformer.fit(tables, data_cache_dir)

        for tn in self.transformer.order:
            table_model_dir = os.path.join(model_cache_dir, tn)
            if not self.model_args[tn].get("synthesize", True):
                continue
            foreign_keys = self.transformer.transformers[tn].config.foreign_keys
            if foreign_keys:
                for i, fk in enumerate(foreign_keys):
                    deg_context, deg = self.transformer.degree_prediction_for(tn, i, data_cache_dir)
                    train_degrees(
                        deg_context, deg, os.path.join(table_model_dir, f"degree{i}"), **self.model_args[tn]["degree"]
                    )

                    isnull = self.transformer.isna_indicator_prediction_for(tn, i, data_cache_dir)
                    if isnull is not None:
                        isna_context, isna = isnull
            else:
                encoded = self.transformer.standalone_encoded_for(tn, data_cache_dir)
                train_standalone(encoded, table_model_dir, **self.model_args[tn]["standalone"])

    def generate(self, out_dir: str, trained_dir: str, table_sizes: Dict[str, int] = {}):
        table_sizes = {t: table_sizes.get(t, self.transformer.fitted_size_of(t)) for t in self.transformer.order}
        self.transformer.prepare_sampled_dir(out_dir)

        for tn in self.transformer.order:
            table_model_dir = os.path.join(trained_dir, tn)
            if self.model_args[tn].get("synthesize", True):
                foreign_keys = self.transformer.transformers[tn].config.foreign_keys
                if foreign_keys:
                    for i, fk in enumerate(foreign_keys):
                        deg_context, _ = self.transformer.degree_prediction_for(tn, i, out_dir)
                        degrees = predict_degrees(
                            deg_context, os.path.join(table_model_dir, f"degree{i}"),
                            expected_sum=table_sizes[tn], tolerance=0 if i > 1 else 0.9,
                            min_val=1 if fk.total_participate else 0,
                            max_val=1 if fk.unique else np.inf
                        )
                        self.transformer.save_degree_for(tn, i, degrees, out_dir)
                else:
                    encoded = generate_standalone(table_sizes[tn], table_model_dir)
                    self.transformer.save_standalone_encoded_for(tn, encoded, out_dir)
            else:
                self.transformer.copy_fitted_for(tn, out_dir)

            self.transformer.prepare_next_for(tn, out_dir)
