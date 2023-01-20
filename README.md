# Node Classification of Hyperparameter optimization with [optuna](https://optuna.org/)

|HyperParameters|min|max|
|-----|---|---|
|learning rate|1e-5|1e-1|
|weight decay|1e-5|1e-1|
|eps|1e-10|1e-6|
|dropout|0.0|0.7|
|number of units|4|512|
|K|5|200|
|alpha|0.05|0.2|

- Dataset
```
Cora, PubMed, CiteSeer
```

- Model
```
APPNP, SplineConv, GAT
```

- Requirements
```
python 3.9.8
torch 1.12.0
torch_geometric 2.0.5
```

- Run all 
```
python main.py --dataset=Cora --model=appnp
python main.py --dataset=Cora --model=splineconv
python main.py --dataset=Cora --model=gat
python main.py --dataset=PubMed --model=appnp
python main.py --dataset=PubMed --model=splineconv
python main.py --dataset=PubMed --model=gat
python main.py --dataset=CiteSeer --model=appnp
python main.py --dataset=CiteSeer --model=splineconv
python main.py --dataset=CiteSeer --model=gat
```