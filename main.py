import argparse
import os
import shutil
import warnings
from typing import Dict

import pandas as pd
import yaml

from irg import TableConfig, IncrementalRelationalGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", default="config/sample.yaml", help="config file, default is a sample"
    )
    parser.add_argument(
        "--input-data-dir", "-i", required=True, help="input data directory, data are in TABLE_NAME.csv"
    )
    parser.add_argument("--output-path", "-o", default="./out", help="output directory")
    return parser.parse_args()


def _validate_data_config(path: str, tables: Dict[str, TableConfig], descr: str):
    for tn, tc in tables.items():
        table = pd.read_csv(os.path.join(path, f"{tn}.csv"))
        if tc.primary_key is not None:
            if table[tc.primary_key].duplicated().any():
                raise ValueError(f"Primary key constraint {tc.primary_key} on {tn} is not fulfilled for {descr}.")
        for fk in tc.foreign_keys:
            parent = pd.read_csv(os.path.join(path, f"{fk.parent_table_name}.csv"))
            fk_str = f"{fk.child_table_name}{fk.child_column_names} -> {fk.parent_table_name}{fk.parent_column_names}"
            if parent[fk.parent_column_names].duplicated().any():
                raise ValueError(f"Foreign key {fk_str} uniqueness on parent is not fulfilled for {descr}.")
            if (parent.merge(
                table, left_on=fk.parent_column_names, right_on=fk.child_column_names, how="outer", indicator="_merged"
            )["_merged"] == "left_only").any():
                raise ValueError(f"Foreign key {fk_str} validity is not fulfilled for {descr}.")


def main():
    warnings.filterwarnings("ignore")
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    tables = {tn: TableConfig.from_dict(ta) for tn, ta in config["tables"].items()}
    config["tables"] = tables

    if os.path.exists(args.output_path):
        shutil.rmtree(args.output_path)
    _validate_data_config(args.input_data_dir, tables, "real")
    synthesizer = IncrementalRelationalGenerator(**config)
    table_paths = {
        tn: os.path.join(args.input_data_dir, f"{tn}.csv") for tn in tables
    }
    synthesizer.fit(table_paths, args.output_path)

    synthesizer.generate(os.path.join(args.output_path, "generated"), os.path.join(args.output_path, "model"))
    _validate_data_config(args.input_data_dir, tables, "synthetic")


if __name__ == '__main__':
    main()
