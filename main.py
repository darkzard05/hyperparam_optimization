import os
import json
import datetime
import argparse

import torch
import torch.nn as nn
from torch import optim
import torch_geometric.transforms as T

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from optuna.trial import TrialState

import nn_model
import load_dataset
from settings import (SUPPORTED_SPLITS, SUPPORTED_DATASETS, SUPPORTED_MODELS,
                      DATA_DEFAULT_PATH, LOG_INTERVAL,
                      EXTRA_MODEL_PARAMS, EXTRA_PARAMS_TYPES)


def preprocess_data(data):
    x = (data.x - data.x.mean()) / data.x.std()
    edge_index, edge_attr = data.edge_index, data.edge_attr
    return x, edge_index, edge_attr


def train(data, model, optimizer, scheduler):
    x, edge_index, edge_attr = preprocess_data(data)
    model.train()
    optimizer.zero_grad()
    output, target = model(x, edge_index, edge_attr)[data.train_mask], data.y[data.train_mask]
    loss = nn.CrossEntropyLoss()(output, target)
    loss.backward()
    optimizer.step()
    scheduler.step()
    return float(loss)


def test(data, model, mask):
    x, edge_index, edge_attr = preprocess_data(data)
    model.eval()
    with torch.no_grad():
        output = model(x, edge_index, edge_attr)
        target = data.y
        pred = output.argmax(dim=1)
        correct = pred[mask] == target[mask]
        accuracy = correct.sum() / mask.sum()
    return float(accuracy)


def evaluate(data, model):
    masks = {'train': data.train_mask, 'val': data.val_mask, 'test': data.test_mask}
    results = {key: test(data, model, mask) for key, mask in masks.items()}
    return results['train'], results['val'], results['test']


def get_common_model_params(trial):
    model_params = {
        'activation': trial.suggest_categorical('activation', ['relu', 'leakyrelu', 'elu']),
        'dropout': trial.suggest_float('dropout', 0.0, 0.7),
        'n_units': trial.suggest_categorical('n_units', [2** i for i in range(2, 8)]),
        'num_layers': trial.suggest_int('num_layers', 1, 5)
        }
    return model_params


def add_extra_model_params(trial, model_name, model_params):
    for param in EXTRA_MODEL_PARAMS[model_name]:
        param_type, param_range = EXTRA_PARAMS_TYPES[param]
        if param_type == 'categorical':
            suggest = trial.suggest_categorical(param, param_range)
        elif param_type == 'int':
            suggest = trial.suggest_int(param, *param_range)
        elif param_type == 'float':
            suggest = trial.suggest_float(param, *param_range)
        model_params[param] = suggest
    return model_params


def get_optim_params(trial):
    optim_params = {
        'lr': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True),
        'optimizer': trial.suggest_categorical('optimizer',
                                           ['Adam', 'NAdam', 'AdamW', 'RAdam']),
        }
    return optim_params


def initialize_model(trial, data, dataset, args):
    model_basic_params = get_common_model_params(trial)
    model_params = add_extra_model_params(trial, args.model, model_basic_params)
    model_params.update({'in_channels': data.num_features,
                         'out_channels': dataset.num_classes
                         })
    model = getattr(nn_model, args.model+'Model')(**model_params)
    
    return model


def initialize_optimizer(trial, model):
    optim_params = get_optim_params(trial)
    optimizer = getattr(optim, optim_params['optimizer'])(model.parameters(),
                                           lr=optim_params['lr'],
                                           weight_decay=optim_params['weight_decay'])
    return optimizer


def objective(trial, data, dataset, args, device):
    model = initialize_model(trial, data, dataset, args)
    optimizer = initialize_optimizer(trial, model)
    model.to(device)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=10, epochs=args.epochs,
                                              anneal_strategy='linear')
    
    best_val_acc, best_test_acc = 0, 0
    
    for epoch in range(1, args.epochs+1):
        loss = train(data, model, optimizer, scheduler)
        train_acc, val_acc, test_acc = evaluate(data, model)
        
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            
        if epoch % LOG_INTERVAL == 0 or best_val_acc == val_acc:
            print(f'epoch: {epoch}, loss: {loss:.3f}, train_acc: {train_acc:.3f}, val_acc: {val_acc:.3f}, test_acc: {test_acc:.3f}')
            
        trial.report(val_acc, epoch)
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return best_test_acc


def save_best_trial_to_json(study, args):
    best_trial = study.best_trial
    
    result = {
        'date_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_name': args.model,
        'dataset_name': args.dataset,
        'dataset_split_type': args.split,
        'best_trials': {
            'params': best_trial.params,
            'value': best_trial.value,
            'number': best_trial.number
            }
        }
    
    filename = f'best_trial_{args.model}_{args.split}_{args.dataset}.json'
    
    with open(filename, 'w') as f:
        json.dump(result, f)


def validate_args(args):
    if args.model not in SUPPORTED_MODELS:
        raise ValueError(f'Model {args.model} is not supported. Supported models: {", ".join(SUPPORTED_MODELS)}.')
    
    if args.dataset not in SUPPORTED_DATASETS:
        raise ValueError(f'Dataset {args.dataset} is not supported. Supported datasets: {", ".join(SUPPORTED_DATASETS)}.')

    if args.split not in SUPPORTED_SPLITS:
        raise ValueError(f'Split type {args.split} is not supported. Supported split types: {", ".join(SUPPORTED_SPLITS)}.')


def display_results(study, args):
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    
    trial = study.best_trial
    
    print(f'''
    Study statistics:
      Number of finished trials: {len(study.trials)}
      Number of pruned trials: {len(pruned_trials)}
      Number of complete trials: {len(complete_trials)}
    Model name: {args.model}
    Dataset name(split type): {args.dataset}({args.split})
    Best trial:
      Value: {trial.value:.4f}
    Parameters:''')
    
    for key, value in trial.params.items():
        value_str = f'{value:.4f}' if isinstance(value, float) else str(value)
        print(f'      {key}: {value_str}')
    

def valid_positive_int(x):
    try:
        x = int(x)
        if x <= 0:
            raise ValueError
        return x
    except ValueError:
        raise argparse.ArgumentTypeError(f'{x} is not a valid positive integer.')


if __name__ == '__main__':
    # settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        help='one of model APPNP, Splineconv, GAT')
    parser.add_argument('--dataset', type=str,
                        help='one of dataset Cora, PubMed, CiteSeer')
    parser.add_argument('--split', type=str, default='public',
                        help='one of dataset split type public, random, full, geom-gcn')
    parser.add_argument('--n_trials', type=valid_positive_int, default=100, help='number of trials')
    parser.add_argument('--epochs', type=valid_positive_int, default=100, help='epochs per trial')
    args = parser.parse_args()
    
    validate_args(args)
    
    dataset_path = os.path.join(DATA_DEFAULT_PATH, args.dataset)
    dataset = load_dataset.load_dataset(path=dataset_path,
                                       name=args.dataset,
                                       split=args.split,
                                       transform=T.TargetIndegree())
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = dataset[0].to(device)
    
    study_name = args.dataset + f'({args.split})' + '_' + args.model + '_study'
    storage_name = 'sqlite:///planetoid-study.db'

    study = optuna.create_study(storage=storage_name,
                                sampler=TPESampler(),
                                pruner=HyperbandPruner(),
                                study_name=study_name,                                
                                direction='maximize',
                                load_if_exists=True)
    study.optimize(lambda trial: objective(trial, data, dataset, args, device), n_trials=args.n_trials)

    save_best_trial_to_json(study, args)
    display_results(study, args)