# Node Classification of Hyperparameter optimization with [optuna](https://optuna.org/)

|dataset|model|Sampler|Pruner|
|-------|-----|-------|------|
|Cora|[APPNP](https://arxiv.org/abs/1810.05997)|TPESampler|HyperBandPruner|

|optimizer|
|-------|
|Adam, NAdam, RAdam, AdamW|

|HyperParameters|min|max|
|-----|---|---|
|learning rate|0|1|
|weight decay|0|0.5|
|eps|0|1|
|dropout|0.1|0.9|
|number of linear layers|1|3|
|number of units|5|200|
|K|1|200|
|alpha|0|0.2|


- Requirements
```
torch, torch_geometric
```
- Run all 
```
python main.py
```
- best accuracy
```
0.85
```
