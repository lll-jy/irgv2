# Prerequisites

Run `make install` to install needed packages and install `irg` package.


# Documentation

Run `make docs` to launch documentation, and visit `localhost:8080` in browser to see the documentation.


# Data Preparation

## Metadata


# Training

## Preparation

1. Make sure the database configuration file is prepared.
1. Make sure the data content of each table is saved in one directory and named nicely.
1. Make sure paths of the cached database directories, log directories, and checkpoint directories exist up to the 
   level of the directories directly holding these directories, and these directories do not contain unwanted 
   and/or un-overridable information. 

## Use Make Script

Run `make train_gpu` to run distributed training on GPUs, and `make train_cpu` to train on CPU.

The following table shows the variables of the script.

| Name | Default | Description | 
|:---|:---|:---|
|`NUM_GPUS`|-|Number of GPUs (only used for distributed training).|
|`PORT`|`1234`|Master port for distributed training (only used for distributed training).|
|`DB_CONFIG`|`config.json`|Path with the configuration of database described as JSON file saved at.|
|`DATA_DIR`|`.`|Path to the directory with all tables, where each table is saved as `TABLE_NAME.csv`.|
|`MTYPE`|`affecting`|Database joining mechanism name.|
|`CACHE_DB`|`real_db`|Path to the directory where cached database with internal structure is saved.|
|`LOG_DIR`|`logs`|Directory saving the tensorboard logging. Tabular models are in `tab/` sub directory and degree models are in `deg/` subdirectory.|
|`CKPT_DIR`|`checkpoints`|Directory saving all checkpoints of trained models. The subdirectory structure is the same as `LOG_DIR`.|

## Use Python Script

The Makefile is equivalent to a simplified version of `Python` script with some features disabled.
To make use the full functionality provided, one may use the `main.py` file directly.

### CLI Arguments and Python Script

In the case that only training is to be executed, `--skip_generate` should be specified in the argument.
Explanation of other arguments can be seen by

    python3 main.py -h
    
Follow the explanation and run the script with desired arguments.

If one wants to run distributed training, run the following script followed by the desired arguments.

    python3 -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=${PORT} main.py

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

The following table shows the variables of the script in addition to training stage.

| Name | Default | Description | 
|:---|:---|:---|
|`OUT_DIR`|`generated`|Output directory of generated tables.|
|`SCALING`|`1`|Scaling factor (uniform, i.e. for all tables).|
|`CACHE_DB_FAKE`|`fake_db`|Fake database cached directory with internal structure for execution.|

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
