# Usage

Some shared or non-temporary configuration can be written to configuration files.
The suggested directories (which already contains some samples) are listed below.

| Argument type                                | Suggested directory | Variable                             | Script argument                                                                                                  |
|----------------------------------------------|---------------------|--------------------------------------|------------------------------------------------------------------------------------------------------------------|
| [trainer argument](#trainer-arguments)       | config/trainer      | `TAB_TARINER_CFG`, `DEG_TRAINER_CFG` | `--default_tab_trainer_args`, `--default_deg_trainer_args`                                                       |
| [train argument](#training-arguments)        | config/train        | `TAB_TRAIN_CFG`, `DEG_TRAIN_CFG`     | `--default_tab_train_args`, `--default_deg_train_args`                                                           |
| [generation argument](#generation-arguments) | -                   | `SCALING`                            | `--default_scaling`, `--scaling`, `--default_gen_tab_bs`, `--gen_tab_bs`, `--default_gen_deg_bs`, `--gen_deg_bs` |
| [evaluator argument](#evaluator-arguments)   | config/evaluator    | `EVALUATOR_CFG`                      | `--evaluator_path`                                                                                               |

In the table above, the variable refers to the variable in `alset.makefile`, 
and the script argument refers to `Python` script `main.py` argument.

In the configuration files, all fields are optional because there is some default settings.

To use the config files in `alset.makefile`, use commands with `_cfg` suffix.

# Trainer Arguments

Some sample trainer argument specification is provided in `config/trainer`.
The config files are JSON files.

## General Settings for All Tabular/Degree Trainers

One can change the trainer model type (`CTGAN`, `TVAE`, or `MLP`) by setting `"trainer_type"` here.
The trainer model type can also be set by `Python` script `--default_tab_trainer_trainer_type` and
`--default_deg_trainer_trainer_type`.
One can also specify other shared arguments for trainer constructors here, which include the following.
For detailed explanation on what these arguments mean, please refer to the API documentation for [`Trainer`](../irg/utils/trainer#irg.utils.trainer.Trainer) constructor.

| Config key      | Script argument for tabular         | Script argument for degree          |
|-----------------|-------------------------------------|-------------------------------------|
| `"distributed"` | `--default_tab_trainer_distributed` | `--default_deg_trainer_distributed` |
| `"autocast"`    | `--default_tab_trainer_autocast`    | `--default_deg_trainer_autocast`    |
| `"log_dir"`     | `--default_tab_trainer_log_dir`     | `--default_deg_trainer_log_dir`     |
| `"ckpt_dir"`    | `--default_tab_trainer_ckpt_dir`    | `--default_deg_trainer_ckpt_dir`    |

Other training settings are specific to the trainer types.
Please refer to the API documentation of the constructor of each type.
Supported types include:

1. [`CTGAN`](../irg/tabular/ctgan#irg.tabular.ctgan.CTGANTrainer).
2. [`TVAE`](../irg/tabular/tvae#irg.tabular.tvae.TVAETrainer).
3. [`MLP`](../irg/tabular/mlp#irg.tabular.mlp.MLPTrainer).

## Hyperparameters for Each Trainer Type

In this section, we list some important hyperparameter settings.
Note that the list below is by no means complete.

### CTGAN Trainer

- `embedding_dim`: Embedding dimensions for latent space.
- `pac`: Set for mitigating mode collapse (if too small, might over-fit; if too large, might under-fit). 
- `gen_optim_lr`: Generator learning rate.
- `disc_optim_lr`: Discriminator learning rate.
- `gen_optim_weight_decay`: Generator weight decay.
- `disc_optim_weight_decay`: Discriminator weight decay.

### TVAE Trainer

- `embedding_dim`: Embedding dimensions for latent space.
- `loss_factor`: Multiplier of reconstruction error.
- `optim_lr`: Learning rate.
- `optim_weight_decay`: Weight decay.

### MLP Trainer

- `hidden_dim`: MLP hidden dimensions (excluding input and output dimensions, as a tuple of integers.)
- `optim_lr`: Learning rate.
- `optim_weight_decay`: Weight decay.

## Special Setting for Specific Trainers

[TODO]

# Training Arguments

Some sample training argument specification is provided in `config/train`.
The config files are JSON files.

## General Settings for All Tabular/Degree Training

One can change the training arguments either by configuration file or by `Python` script CLI arguments.
This part corresponds to the [`Trainer.train`](../irg/utils/trainer#irg.utils.trainer.Trainer.train) method 
(other than known and unknown input tensors).
Accepted settings are shown as follows.

| Config key     | Script argument for tabular      | Script argument for degree       |
|----------------|----------------------------------|----------------------------------|
| `"epochs"`     | `--default_tab_train_epochs`     | `--default_deg_train_epochs`     |
| `"batch_size"` | `--default_tab_train_batch_size` | `--default_deg_train_batch_size` |
| `"shuffle"`    | `--default_tab_train_shuffle`    | `--default_deg_train_shuffle`    |
| `"save_freq"`  | `--default_tab_train_save_freq`  | `--default_deg_train_save_freq`  |
| `"resume"`     | `--default_tab_train_resume`     | `--default_deg_train_resume`     |

## Special Setting for Specific Training Tasks

[TODO]

# Generation Arguments

## Scaling

### Uniform Scaling

One can set `SCALING` variable in `alset.makefile` or `--default_scaling` argument in `Python` script CLI argument
for uniform scaling.
For example, if we want to generate a database twice the size of the original one, we set the number to 2.

### Non-uniform Scaling

This is not prepared in `alset.makefile`. 
One can set by `--scaling` following instruction on `python3 main.py train_gen -h`.

## Batch sizes

By similar manners as scaling factors, one can set the generation batch sizes for tabular and degree models.
This flexibility is not opened in `alset.makefile`, 
and aim only for full resource usage in training for better efficiency.

# Evaluator Arguments

## Writing Config File

Some sample evaluator argument specification is provided in `config/evaluator`.
Note that the given ones are for reference purpose only and may not be complete.
The config files are JSON files.
Allowed settings are the arguments for 
[`SyntheticDatabaseEvaluator`](../irg/metrics/evaluator#irg.metrics.evaluator.SyntheticDatabaseEvaluator) constructor.

To set configuration with long specification such as SQL queries, one can write the content in a draft first and paste 
it nicely in JSON format back (since JSON format may not be very readable in such circumstances).

Two special large configuration settings are `"default_args"`, for default setting, and `"tabular_args"`, for settings
for each particular tabular dataset for evaluation.
They are arguments to 
[`SyntheticTableEvaluator`](../irg/metrics/tabular/evaluator#irg.metrics.tabular.evaluator.SyntheticTableEvaluator).
In particular, `"default_args"` are raw arguments that apply to all tabular data evaluation in this database.
`"tabular_args"` is a dictionary of table type mapped to a dictionary mapping each table name and/or description to its 
specific arguments.

## Python Script Arguments

One can also override content from the configuration file by `Python` script CLI arguments.
For details please see the instructions by `python3 main.py evaluate -h`.

Those large configurations would be read in separate files
whose path are specified in CLI argument with `_from_file` suffix.
Supported such arguments include `parent_child_pairs`, `queries`, `query_args`, `tabular_args`, and `default_args`.

## Guidelines

For each specific database, it is suggested to write evaluator configurations separately,
because each database have different properties that we may specifically want the synthetic version to satisfy.
We give a list of such settings for reference as follows.

1. **Parent-child pairs**. If we want to evaluate the result of joining two non-adjacent tables in database hierarchy,
   we can use this setting. The key for this setting is `"parent_child_pairs"`, and one can specify the pairs of tables 
   want to be joined by giving the list of valid foreign key references (that is, the two tables and columns specified
   must be join-able). Format for a foreign key is specified for 
   [`ForeignKey`](../irg/schema/database/base#irg.schema.database.base.ForeignKey.dict).
   Different databases have different tables, so this should be set every time working on a new database.
2. **Queries**. If we want to construct specific tables constructed by some queries in the database, we would need to 
   specify what the queries are, which cannot be shared across different databases.
   One should provide the queries as a dictionary of short description mapped to the SQL query.
   The key for this setting is `"queries"`.
3. **Invalid combinations per table**. Some tables (or tables constructed by queries) in the database should satisfy 
   some specific rules. For example, residency status in Singapore being a citizen means that the student's country of
   citizenship cannot be anything other than Singapore. If we want to check on such rules, we can specify in 
   `"tabular_args"` -> `"tables"` (or other table types such as `"parent child"`) -> `"eval_invalid_comb"` as `true` and 
   `"invalid_comb"` as a dictionary of short description mapped to a tuple of list of corresponding column names and 
   list of tuples of invalid combination values.
4. **Specific ML tasks**. We may want to specify specific ML tasks for some tables (or tables constructed by queries) 
   in the database. In `"tabular_args"` -> `"tables"` (again, or other types) -> `"reg_tasks"` for regression tasks, and
   `"clf_tasks"` for classification tasks, give a dictionary with key as short description of a specific task in that 
   table, and tuple of (target name, feature names) as value specifying what this task is exactly.
5. **Degree count**. Sometimes we want to check how many times one specific entity appear in some tables (or tables 
   constructed by queries) and recover this distribution. For example, we may want to check how many modules a student
   takes. We can evaluate on this distribution by setting `"tabular_args"` -> `"tables"` (or other types) -> 
   `"count_on"` as a dictionary mapping a short description of the degree checking to a list of column names to count 
   on. For example, counting modules of a student can be `"mod_per_st": ["student_token"]` in table for module 
   enrolment.
