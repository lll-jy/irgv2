import os
from typing import Any, Dict, List

from .schema import RelationalTransformer, TableConfig
from .standalone import train_standalone, generate_standalone


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
            if self.transformer.transformers[tn].config.foreign_keys:
                pass
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
                    pass
                else:
                    encoded = generate_standalone(table_sizes[tn], table_model_dir)
                    self.transformer.save_standalone_encoded_for(tn, encoded, out_dir)
            else:
                self.transformer.copy_fitted_for(tn, out_dir)

            self.transformer.prepare_next_for(tn, out_dir)
