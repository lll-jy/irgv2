# Incremental Relational Generator

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