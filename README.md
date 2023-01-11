# Node Classification of Hyperparameter optimization with [optuna](https://optuna.org/)

|HyperParameters|min|max|
|-----|---|---|
|learning rate|1e-5|1e-1|
|weight decay|1e-5|1e-1|
|eps|1e-10|1e-6|
|dropout|0.0|1.0|
|number of units|4|1024|
|K|5|200|
|alpha|0|0.5|

- Dataset
```
Cora, PubMed, CiteSeer
```

- Model
```
APPNP, SplineConv
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
python main.py --dataset=PubMed --model=appnp
python main.py --dataset=PubMed --model=splineconv
python main.py --dataset=CiteSeer --model=appnp
python main.py --dataset=CiteSeer --model=splineconv
```
- best accuracy
```
Cora: 0.85
PubMed: 0.8
CiteSeer: 0.73
```
