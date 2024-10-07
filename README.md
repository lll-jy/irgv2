# Incremental Relational Generator (IRG)

## Pre-requisites

1. `Python>=3.10`.
2. Dependencies `pip install -r requirements.txt`.

## Important Notes

The project is now in commercial usage, so the full code is not disclosed. This repository aims to provide
the code skeleton and reproduces the main logic for IRG. It is still executable, but some functions are filled with 
placeholders that do not do the same thing as what is described in the paper, and some core parameters are not exposed.

All core methods and functions in the codebase have docstrings describing what they do. Methods that are filled by 
placeholders are annotated with `@placeholder`. 

The evaluation framework is also not exposed for the same reason.

## Quick Start

To train and generate a synthetic database, run the following command:

```shell
python main.py -c CONFIG_FILE -i DATA_DIR -o OUT_DIR
```

where `CONFIG_FILE` is the configuration yaml file, with a sample in `config/sample.yaml`, `DATA_DIR` is the directory
where real data is stored, and each table name mentioned in the config should have the file `TABLE_NAME.csv` found
in this directory, and `OUT_DIR` be the directory where trained results can be found.

The generated synthetic data can be found in `OUT_DIR/generated` csv files.
