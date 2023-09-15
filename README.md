<!-- # Node Classification of Hyperparameter optimization with [optuna](https://optuna.org/)

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
python main.py --dataset=Cora --model=Splineconv
python main.py --dataset=Cora --model=GAT
python main.py --dataset=PubMed --model=APPNP
python main.py --dataset=PubMed --model=Splineconv
python main.py --dataset=PubMed --model=GAT
python main.py --dataset=CiteSeer --model=APPNP
python main.py --dataset=CiteSeer --model=Splineconv
python main.py --dataset=CiteSeer --model=GAT

$ pip install optuna-dashboard
$ optuna-dashboard sqlite://planetoid-study.db
``` -->

# Graph Neural Network Hyperparameter Optimization

This project focuses on tuning the hyperparameters of various Graph Neural Network (GNN) models and provides implementations for the APPNP, Splineconv, and GAT models using [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/).

## Overview

- **Hyperparameter Optimization**: Using Bayesian optimization provided by Optuna to find the best hyperparameters that lead to optimal model performance.

- **Model Implementations**: Built-in support for APPNP, Splineconv, and GAT architectures. Each model has its own customizable parameters.

## Features

- Model-specific hyperparameter search space.
- Modular design for easy expansion to new GNN architectures.
- Pre-defined activation functions for flexibility.
- Comprehensive evaluation on graph datasets like Cora, PubMed, and CiteSeer.

## Requirements

- Python 3.x
- torch
- torch_geometric
- optuna

## Usage

1. Clone the repository:
```
git clone https://github.com/darkzard05/planetoid_search_hyperparam.git

```
2. Install the requied packages:
```
pip install -r requirements.txt
```
3. Run the main script with desired arguments:
```
python main.py --dataset=Cora --model==APPNP --split=public --n_trials=100 --epochs=100
```

## Arguments
- model: Model to be used (APPNP, Splineconv, GAT). (Default: APPNP)
- dataset: Dataset to be used (Cora, PubMed, CiteSeer). (Default: Cora)
- split: Dataset split type (public, random, full, geom-gcn). (Default: public)
- n_trials: Number of trials. (Default: 100)
- epochs: Number of epochs per trial. (Default:100)

## Models

### [APPNP](https://arxiv.org/abs/1810.05997)
- Implements the Approximated Personalized Propagation of Neural Predictions (APPNP) layer.
### [Splineconv](https://arxiv.org/abs/1711.08920)
- Utilizes B-spline basis functions to hierarchically partition and transform the input graph.
### [GAT](https://arxiv.org/abs/1710.10903)
- Graph Attention Networks (GAT) use attention mechanisms to weigh neighbor features.
