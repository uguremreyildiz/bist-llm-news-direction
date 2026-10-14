# Model selection protocol and hyperparameter search space

This document records exactly how the reported configuration was selected, so that the choice
can be audited without re-running the benchmark. The selected model itself is reproduced by
`reproduce.py`.

## 1. Configuration grid

152 configurations were evaluated, formed by crossing:

| Axis | Levels |
|---|---|
| Feature set | `M0_Close`, `M1_Close+Macro`, `M2_Emb`, `M3_Full` |
| Text encoder | `bge_m3` (1024-d), `bert_tr`, `standard_tr`, `finturkbert`, `mpnet_multi` (768-d), `fasttext` (300-d) |
| Reducer | PCA, UMAP, autoencoder |
| Classifier | logistic regression, Gaussian naive Bayes, LightGBM, random forest |

`M0_Close` and `M1_Close+Macro` contain no text features, so the encoder and reducer axes do not
apply to them; the grid is therefore not the full Cartesian product.

## 2. Fold structure

Expanding window over 393 weeks: initial training block 156 weeks, test block 12 weeks, step 12
weeks. The first 60% of the sample forms the tuning region and the remainder the evaluation
region; one fold straddling the boundary is discarded. This yields **6 inner folds** used for
hyperparameter search and **12 outer folds** used for evaluation. No out-of-sample week
influences any hyperparameter, encoder, reducer or classifier choice.

## 3. Search

* Sampler: Optuna TPE (`TPESampler`), `seed = 42`.
* Budget: **50 trials per configuration**.
* Objective: mean macro-F1 across the 6 inner folds (maximised).
* Logistic regression, LightGBM and random forest use `class_weight="balanced"`; within each trial and each fold, the reducer
  and `StandardScaler` are fitted on that fold's training rows only.

### Classifier search spaces

**Logistic regression**

| Parameter | Range |
|---|---|
| `C` | 1e-3 .. 1e2, log scale |
| `penalty` | `l1`, `l2` (solver `liblinear`) |

**Gaussian naive Bayes**

| Parameter | Range |
|---|---|
| `var_smoothing` | 1e-12 .. 1e-2, log scale |

**LightGBM**

| Parameter | Range |
|---|---|
| `n_estimators` | 100 .. 400 |
| `learning_rate` | 0.01 .. 0.2, log scale |
| `num_leaves` | 15 .. 127 |
| `max_depth` | 3 .. 10 |
| `min_child_samples` | 10 .. 80 |
| `subsample` | 0.6 .. 1.0 |
| `colsample_bytree` | 0.6 .. 1.0 |
| `reg_lambda` | 1e-3 .. 10.0, log scale |

**Random forest**

| Parameter | Range |
|---|---|
| `n_estimators` | 100 .. 400 |
| `max_depth` | 3 .. 20 |
| `min_samples_leaf` | 1 .. 30 |
| `max_features` | `sqrt`, `log2`, 0.5 |

### Reducer settings

The reducers are fixed rather than tuned, so that differences across configurations are
attributable to the encoder and classifier rather than to unequal search effort:

| Reducer | Settings |
|---|---|
| PCA | `n_components = 32` per stream, `random_state = 42` |
| UMAP | `n_components = 32`, `n_neighbors = 15`, `min_dist = 0.1` |
| Autoencoder | bottleneck 32, hidden layer 256, 40 epochs, learning rate 1e-3, batch size 256 |

Reduction is applied separately to the stock-news and macro-news embedding blocks, giving
32 + 32 = 64 reduced features.

## 4. Selected configuration

`M3_Full / bge_m3 / PCA / logistic regression`, with `C = 0.02005544632966435` and
`penalty = "l1"` — the highest mean inner-fold macro-F1 in the grid. These values are hard-coded
in `reproduce.py`, which then evaluates the model on the 12 held-out folds.
