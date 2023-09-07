# Node Classification of Hyperparameter optimization with [optuna](https://optuna.org/)

|Hyperparameter|value|
|-----|-----|
|learning rate|(1e-5, 1e-1)|
|weight decay|(1e-5, 1e-1)|
|dropout|(0.0, 0.7)|
|number of units|(4, 512)|
|K|(5, 200)|
|alpha|(0.05, 0.2)|
|kernel size|(1, 8)|
|heads|(1, 8)|
|optimization|Adam, NAdam, AdamW, RAdam|
|activation|ReLU, LeakyReLU, ELU|

- Dataset(split type)
```
Cora, PubMed, CiteSeer
(public, random, full, geom-gcn)
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
python main.py --dataset=Cora --model=APPNP
python main.py --dataset=Cora --model=splineconv
python main.py --dataset=Cora --model=GAT
python main.py --dataset=PubMed --model=APPNP
python main.py --dataset=PubMed --model=splineconv
python main.py --dataset=PubMed --model=GAT
python main.py --dataset=CiteSeer --model=APPNP
python main.py --dataset=CiteSeer --model=splineconv
python main.py --dataset=CiteSeer --model=GAT

$ pip install optuna-dashboard
$ optuna-dashboard sqlite://planetoid-study.db
```