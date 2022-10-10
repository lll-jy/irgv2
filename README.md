# Incremental Relational Generator

## Performance

((need to find some values that produce reasonable results))

Default setting:
- **batch size: 100**
- **epochs: 50**
- **learning rate: 0.0005**
- **weight decay: 0.05**

Default setting for CTGAN:
- **embedding dimension: 20**
- **generator dimension: [64, 64]**
- **discriminator dimension: [64, 64]**
- **pac: 20**
- **discriminator step: 1**

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

