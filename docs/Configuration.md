# Usage

Some shared or non-temporary configuration can be written to configuration files.
The suggested directories (which already contains some samples) are listed below.

| Argument type                          | Suggested directory | Variable                             | Script argument                                            |
|----------------------------------------|---------------------|--------------------------------------|------------------------------------------------------------|
| [trainer argument](#trainer-arguments) | config/trainer      | `TAB_TARINER_CFG`, `DEG_TRAINER_CFG` | `--default_tab_trainer_args`, `--default_deg_trainer_args` |
| [train argument](#training-arguments)  | config/train        | `TAB_TRAIN_CFG`, `DEG_TRAIN_CFG`     | `--default_tab_train_args`, `--default_deg_train_args`     |

In the table above, the variable refers to the variable in `alset.makefile`, 
and the script argument refers to `Python` script `main.py` argument.
Some argument type may have multiple variables and script arguments, referring to the setting for tabular and degree 
models respectively.

In the configuration files, all fields are optional because there is some default settings.

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

# Evaluation Arguments
