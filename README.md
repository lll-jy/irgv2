# Incremental Relational Generator

## Performance

((need to find some values that produce reasonable results))

Default setting:
- **batch size:**
- **epochs:**
- **learning rate:**
- **weight decay:**

Default setting for CTGAN:
- **embedding dimension:**
- **generator dimension:**
- **discriminator dimension:**
- **pac:**
- **discriminator step:**

Default setting for TVAE:
- **embedding dimension:**
- **compress dimension:**
- **decompress dimension:**
- **loss factor:**

### ALSET

Compare the effect following (italic font means default setting):

- `mtype`: unrelated (ur), parent-child (pc), ancestor-descendant (ad), _affecting_ (af).
- `tab model`: _CTGAN_, TVAE, MLP.
- `deg model`: CTGAN, TVAE, _MLP_.

((default argument != default setting marked. all on all tables, complete tables.))

The result is shown below.

| mtype | tab   | deg   | perf |
|-------|-------|-------|------|
| ur    | CTGAN | MLP   |      |
| pc    | CTGAN | MLP   |      |
| ad    | CTGAN | MLP   |      |
| af    | CTGAN | MLP   |      |
| af    | TVAE  | MLP   |      |
| af    | MLP   | MLP   |      |
| af    | CTGAN | CTGAN |      |
| af    | CTGAN | TVAE  |      |

((remember to give specific configuration setup for evaluation))

Comparing out model to baselines, the result is shown below.

| model  | perf |
|--------|------|
| IRGAN  |      |
| SDV    |      |
| CTGAN1 |      |
| CTGAN2 |      |

## [User Guide](./docs/UserGuide.md)


{
  "epochs": 50,
  "batch_size": 100,
  "shuffle": true,
  "save_freq": 10,
  "resume": true
}

{
  "trainer_type": "CTGAN",
  "embedding_dim": 20,
  "generator_dim": [64, 64],
  "discriminator_dim": [64, 64],
  "pac": 20,
  "discriminator_step": 1,
  "gen_optimizer": "AdamW",
  "gen_scheduler": "StepLR",
  "disc_optimizer": "AdamW",
  "disc_scheduler": "StepLR",
  "gen_optim_lr": 0.0005,
  "disc_optim_lr": 0.0005,
  "gen_optim_weight_decay":0.05,
  "disc_optim_weight_decay":0.05
}