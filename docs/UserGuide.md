# Prerequisites

Run `make install` to install needed packages and install `irg` package.

If this is already run on the device, one can save time by updating instead of re-installing by `make update`.


# Documentation

Run `make docs` to launch documentation, and visit `localhost:8080` in browser to see the documentation.

Or one can set `DOC_PORT` argument for `make docs` and visit `localhost` at this port.


# Data Preparation

For a detailed and elaborated tutorial on data preparation, please go to [`preparation`](./preparation).

## Pre-defined Datasets

### Use Make Script

Run `make prepare` to prepare files for training on pre-defined database datasets.

For convenience, we copy the relevant script here.

    python3 process.py database ${DB_NAME} \
        --src_data_dir ${SRC_DATA_DIR} \
        --data_dir ${DATA_OUTPUT_DIR}/data \
        --meta_dir ${DATA_OUTPUT_DIR}/metadata \
        --out ${DATA_OUTPUT_DIR}/db_config.json \
        --redo_meta \
        --redo_data

#### Variables

The following table shows the variables of the script.

| Name | Default | Description | 
|:---|:---|:---|
|`DB_NAME`|`rtd`|Name of predefined database.|
|`SRC_DATA_DIR`|`src`|Path of directory holding the source data files.|
|`DATA_OUTPUT_DIR`|`data`|Path of directory to save processed data and metadata files to.|

#### Input Structure

Initial data saved at `SRC_DATA_DIR`.

#### Output Structure

The structure is described in YAML-like format for readability.
Everything inside `{}` can be changed according to actual need.
Capitalized directory names are the variables.
JSON files' content may also be desribed in YAML-like format if elaboration is needed there.
This format applies to all structure description in this guide.

    DATA_OUTPUT_DIR:
      db_config.json: "JSON file with database configuration"
      data:
        - {TABLE1_NAME}.csv
        - {TABLE2_NAME}.csv
        - ...
      metadata:
        - {TABLE1_NAME}.json
        - {TABLE2_NAME}.json
        - ...

### Use Python Script

In `examples` package, there are processing codes for a set of pre-defined databases.
To generate required files for a database, one can simply call `python3 process.py database {DATABASE_NAME}`.
To see detailed explanations on command line arguments, one can run `python3 process.py -h`.

## Custom Datasets

One can see `examples` package for demonstration of how to process datasets, and possibly
add processing code for their own databases in this package.


# Training

## Preparation

1. Make sure the database configuration file is prepared.
1. Make sure the data content of each table is saved in one directory and named nicely.
1. Make sure paths of the cached database directories, log directories, and checkpoint directories exist up to the 
   level of the directories directly holding these directories, and these directories do not contain unwanted 
   and/or un-overridable information. 

## Use Make Script

Run `make train_gpu` to run distributed training on GPUs, and `make train_cpu` to train on CPU.

For convenience, we copy the relevant script (CPU) here.

    python3 -W ignore main.py --log_level ${LOG_LEVEL} train_gen \
        --db_config_path ${DATA_OUTPUT_DIR}/db_config.json \
        --data_dir ${DATA_OUTPUT_DIR}/data \
        --mtype ${MTYPE} \
        --db_dir_path ${MODEL_OUTPUT_DIR}/real_db \
        --aug_resume \
        --default_tab_trainer_log_dir ${MODEL_OUTPUT_DIR}/tf/tab \
        --default_deg_trainer_log_dir ${MODEL_OUTPUT_DIR}/tf/deg \
        --default_tab_trainer_ckpt_dir ${MODEL_OUTPUT_DIR}/ckpt/tab \
        --default_deg_trainer_ckpt_dir ${MODEL_OUTPUT_DIR}/ckpt/deg \
        --skip_generate

### Variables

The following table shows the variables of the script.

| Name | Default | Description | 
|:---|:---|:---|
|`NUM_GPUS`|-|Number of GPUs (only used for distributed training).|
|`PORT`|`1234`|Master port for distributed training (only used for distributed training).|
|`DATA_OUTPUT_DIR`|`data`|Directory holding data of the database.|
|`MTYPE`|`affecting`|Database joining mechanism name.|
|`MODEL_OUTPUT_DIR`|`output`|Directory saving model outputs.|

### Input Structure

`DATA_OUTPUT_DIR` from data processing.

### Output Structure

    MODEL_OUTPUT_DIR:
      real_db:
        - config.json: 
            primary_keys: "Input primary_keys for Database constructor"
            foreign_keys: "Input foreign_keys for Database constructor"
            data_dir: "Input data_dir for Database constructor"
            order: "Order of tables by name"
        - {TABLE1_NAME}.pkl: "Table object"
        - {TABLE2_NAME}.pkl
        - ...
      tf:
        tab: "tabuar models"
          - {TABLE1_NAME}:
            - events.out.tfevents...
            - events.out.tfevents...
          - {TABLE2_NAME}:
            - ...
          - ...
        deg:
          - {TABLE2_NAME}:
            - events.out.tfevents...
            - events.out.tfevents...
          - ...
      ckpt:
        tab:
          - {TABLE1_NAME}:
            - epoch_0000001.pt: "Checkpoint after 1 epoch"
            - epoch_0000002.pt: "Checkpoint after 2 epoch"
            - ...
            - step_0000001.pt: "Checkpoint after 1*save_freq steps"
            - step_0000002.pt: "Checkpoint after 2*save_freq steps"
            - ...
          - {TABLE2_NAME}:
            - ...
          - ...
        deg:
          - {TABLE2_NAME}:
            - ...
          - ...

## Use Python Script

The Makefile is equivalent to a simplified version of `Python` script with some features disabled.
To make use the full functionality provided, one may use the `main.py` file directly.

### CLI Arguments and Python Script

In the case that only training is to be executed, `--skip_generate` should be specified in the argument.
Explanation of other arguments can be seen by `python3 main.py -h`.
    
Follow the explanation and run the script with desired arguments.

If one wants to run distributed training, run the following script followed by the desired arguments.

    python3 -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=${PORT} main.py train_gen

### Default and Overriding Arguments

Here we highlight the usage of train and trainer arguments.
There are for groups of arguments handled in the same manner: `tab_trainer`, `deg_trainer`, `tab_train`,
and `deg_train`, where `tab` stands for tabular model, `deg` stands for degree model, and `trainer` stands for arguments
to `Trainer` constructor and `train` stands for arguments to `.train` method of `Trainer`.

In each group (`GROUP`), there are mainly three types of arguments:

- `--default_GROUP_args`: path to the JSON file where a default argument set is specified.
- `--GROUP_args`: path to the JSON file where a `dict` of table name mapped to its own argument set is specified, and
  its content should be a `dict` of `str` mapped to something with similar structure as the content of the file in 
  the argument above.
- `--default_GROUP_ARG`: some (only a subset) arguments to override default settings in the file by CLI arguments; 
  namely, if an argument is specified in both default file and CLI argument, the CLI value will be used.

Arguments overridable from CLI include the following:

- `trainer`: `trainer_type`, `distributed`, `autocast`, `log_dir`, `ckpt_dir`.
- `train`: `epochs`, `batch_size`, `shuffle`, `save_freq`, `resume`.


# Generation

## Preparation

1. Make sure checkpoints and cached database from training stage are saved nicely.
1. Make sure paths of the output directories and cached fake database directories also exist up to the level of the 
   directories directly holding these directories, and these directories do not contain unwanted and/or un-overridable 
   information.

## Use Make Script 

Run `make generate_gpu` to run distributed generation on GPUs, and `make generate_cpu` to generate on CPU.

For convenience, we copy the relevant script (CPU) here.

	python3 -W ignore main.py --log_level ${LOG_LEVEL} train_gen \
        --distrubted \
        --db_config_path ${DB_CONFIG} \
        --data_dir ${DATA_DIR} \
        --mtype ${MTYPE} \
        --db_dir_path ${MODEL_OUTPUT_DIR}/real_db \
        --aug_resume \
        --skip_train \
        --default_tab_train_resume \
        --default_deg_train_resume \
        --default_tab_trainer_log_dir ${MODEL_OUTPUT_DIR}/tf/tab \
        --default_deg_trainer_log_dir ${MODEL_OUTPUT_DIR}/tf/deg \
        --default_tab_trainer_ckpt_dir ${MODEL_OUTPUT_DIR}/ckpt/tab \
        --default_deg_trainer_ckpt_dir ${MODEL_OUTPUT_DIR}/ckpt/deg \
        --save_generated_to ${OUT_DIR} \
        --default_scaling ${SCALING} \
        --save_synth_db ${MODEL_OUTPUT_DIR}/fake_db


### Variables

The following table shows the variables of the script in addition to training stage.

| Name | Default | Description | 
|:---|:---|:---|
|`GENERATE_OUTPUT_DIR`|`generated`|Output directory of generated output.|
|`SCALING`|`1`|Scaling factor (uniform, i.e. for all tables).|

### Input Structure

`MODEL_OUTPUT_DIR` from training stage.

### Output Structure

    GENERATE_OUTPUT_DIR:
      generated: "Same structure as DATA_OUTPUT_DIR/data but for synthetic version"
      fake_db: "Same structure as MODEL_OUTPUT_DIR/real_db but for synthetic version"
    

## Use Python Script

### CLI Arguments and Python Script

In the case that only generation is to be executed, `--skip_train` should be specified in the argument.
Yet indeed, training and generation can be executed together.

To continue from training, `--aug_resume`, `--default_tab_train_resume`, `--default_deg_train_resume` should be set,
and inputs to `ckpt_dir` for trainer constructors need to hold the checkpoints wanted.

### Default and Overriding Arguments

There are three arguments supporting table-wise specification, where each is a built-in data type (`float` or `int`) 
in this case. They are `scaling`, `gen_tab_bs`, and `gen_deg_bs`.
For each of the argument one can specify a default value under `--default_ARG`.
To specify specific values for tables, one can use `--ARG` followed by 2-tuples of table name and expected value.
For more details, one can consult `python3 main.py -h`.


# Evaluation

## Preparation

1. Make sure real database is saved by calling `save_to_dir` to some path.
1. Make sure all synthetic databases for evaluation are saved by calling `save_to_dir` to some path.
1. Prepare configuration files for evaluation if needed.

## Use Make Script

Because of simpliciation in `Makefile`, please make sure all synthetic databases are saved to directories with names
well-describes this synthetic database version (path to the directory does not matter).

Run `make evaluate` to execute evaluation.

For convenience, we copy the relevant script here.

	python3 -W ignore main.py --log_level ${LOG_LEVEL} evaluate \
        --real_db_dir ${MODEL_OUTPUT_DIR}/real_db \
        --fake_db_dir ${FAKE_DB} \
        --evaluator_path ${EVAL_CONFIG}/constructor.json \
        --evaluate_path ${EVAL_CONFIG}/evaluate.json \
        --save_eval_res_to ${EVAL_OUTPUT_DIR}/trivial \
        --save_complete_result_to ${EVAL_OUTPUT_DIR}/complete \
        --save_synthetic_tables_to ${EVAL_OUTPUT_DIR}/tables/synthetic \
        --save_tables_to ${EVAL_OUTPUT_DIR}/tables/real \
        --save_visualization_to ${EVAL_OUTPUT_DIR}/visualization \
        --save_all_res_to ${EVAL_OUTPUT_DIR}/result

### Variables
 
The following table shows the variables of the script.

| Name | Default | Description | 
|:---|:---|:---|
|`MODEL_OUTPUT_DIR`|`output`|Training output directory.|
|`FAKE_DB`|`fake_db`|Fake databases saved directory path (from `GENERATE_OUTPUT_DIR/fake_db` of generation). If multiple synthetic databases (based on the same real database) are to be evaluated, one can specify this variable as `CACHE_DB_FAKE=(DB1 DB2 DB3)`.|
|`EVAL_CONFIG`|`eval_conf`|Path to configurations for evaluation. Should contain two files: `constructor.json` and `evaluate.json`.|
|`EVAL_GENERATE_OUTPUT_DIR`|`evaluation`|Path to save the evaluation output.|

### Input Structure

    MODEL_OUTPUT_DIR: "Result from training"
    FAKE_DB: "Result from generation, but can be a list"
    EVAL_CONFIG:
      - constructor.json: "Arguments to SyntheticDatabaseEvaluator.__init__"
      - evaluate.json: "Arguments to SyntheticDatabaseEvaluator.evaluate"

### Output Structure

    EVAL_GENERATE_OUTPUT_DIR:
      - trivial
        - {SYNTHETIC1}
          - tables
            - {TABLE1_NAME}
              - corr
                - {TABLE1_NAME}.png: "Correlation matrics of real and fake tables"
              - clf
                - {TABLE1_NAME}.csv: "DataFrame describing evaluation result for all experiments done, where each row correspond to a task and each column correspond to a model"
              - reg
                - {TABLE1_NAME}.csv: "Same as clf but for regression tasks"
            - {TABLE2_NAME}
              - ...
            - ...
          - "parent child"
            - "0__{FK1_CHILD}__{FK1_PARENT}": "Same structure as subdirectories in tables"
            - "1__{FK2_CHILD}__{FK2_PARENT}"
            - ...
          - joined
            - joined
          - queries
            - {QUERY1_DESCR}
            - {QUERY2_DESCR}
            - ...
      - complete: 
        - {SYNTHETIC1}: "A dict with the same structure as trivial directory, and content of each 'subdirectory' is the full evaluation results described as Series"
        - {SYNTHETIC2}: ...
        - ...
      - tables
        - real
          - tables
            - {TABLE1_NAME}.pkl: "Table object"
            - {TABLE2_NAME}.pkl
            - ...
          - ...
        - synthetic
          - {SYNTHETIC1}: "Same as real under parent directory"
          - {SYNTHETIC2}
          - ...
      - visualization
        - tables
          - {TABLE1_NAME}
            - {SYNTHETIC1}:
              - {VIS1}.png: "Visualization of the table by a pairplot of the table with dimension reduced"
              - {VIS2}.png
              - ...
            - {SYNTHETIC2}
            - ...
          - ...
        - ...
      - result
        - result.pt: "Dictionary of synthetic versions to the evaluation result on this database as a DataFrame"

## Use Python Script

Run with `python3 main.py evaluate` followed by command line arguments. 
For more details, one can consult `python3 main.py -h`.
