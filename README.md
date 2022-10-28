# Search Hyperparameters with [optuna](https://optuna.org/)

|dataset|model|Sampler|Pruner|
|-------|-----|-------|------|
|Cora|[APPNP](https://arxiv.org/abs/1810.05997)|TPESampler|HyperBandPruner|

|optimizer|
|-------|
|Adam, NAdam, RAdam, AdamW|

|HyperParameters|min|max|
|-----|---|---|
|learning rate|0|1|
|weight decay|0|1|
|eps|0|1|
|dropout|0.1|0.5|
|number of linear layers|1|3|
|number of units|5|200|
|K|1|200|
|alpha|0|1|



* best accuracy 0.85