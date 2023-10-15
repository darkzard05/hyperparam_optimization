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
from settings import (SUPPORTED_MODELS, SUPPORTED_DATASETS, SUPPORTED_SPLITS,
                      DATA_DEFAULT_PATH, LOG_INTERVAL,
                      EXTRA_MODEL_PARAMS, EXTRA_PARAMS_TYPES,
                      )


def preprocess_data(data):
    x = (data.x - data.x.mean()) / data.x.std()
    edge_index = data.edge_index
    edge_attr = data.edge_attr
    return x, edge_index, edge_attr


def train(x, edge_index, edge_attr, data, model, optimizer):
    model.train()
    optimizer.zero_grad()
    output, target = model(x, edge_index, edge_attr)[data.train_mask], data.y[data.train_mask]
    loss = nn.CrossEntropyLoss()(output, target)
    loss.backward()
    optimizer.step()
    return float(loss)


def test(x, edge_index, edge_attr, data, model, mask):
    model.eval()
    with torch.no_grad():
        output = model(x, edge_index, edge_attr)
        target = data.y
        pred = output.argmax(dim=1)
        correct = pred[mask] == target[mask]
        accuracy = correct.sum() / mask.sum()
    return float(accuracy)


def evaluate(x, edge_index, edge_attr, data, model):
    masks = {'train': data.train_mask, 'val': data.val_mask, 'test': data.test_mask}
    results = {key: test(x, edge_index, edge_attr, data, model, mask) for key, mask in masks.items()}
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


def initialize_model_and_optimizer(trial, data, dataset, args):
    # Get model parameters
    model_basic_params = get_common_model_params(trial)
    model_extra_params = add_extra_model_params(trial, args.model, model_basic_params)
    model_params = {'in_channels': data.num_features,
                    'out_channels': dataset.num_classes,
                    **model_basic_params, **model_extra_params}
    model_class_name = args.model+'Model'
    model = getattr(nn_model, model_class_name)(**model_params)
    
    # Get optimizer parameters
    optim_params = get_optim_params(trial)
    optimizer = getattr(optim, optim_params['optimizer'])(model.parameters(),
                                           lr=optim_params['lr'],
                                           weight_decay=optim_params['weight_decay'])
    return model, optimizer


def objective(trial, data, dataset, args, device):
    model, optimizer = initialize_model_and_optimizer(trial, data, dataset, args)
    model.to(device)
    
    best_val_acc, best_test_acc = 0, 0
    for epoch in range(1, args.epochs+1):
        loss = train(x, edge_index, edge_attr, data, model, optimizer)
        train_acc, val_acc, test_acc = evaluate(x, edge_index, edge_attr, data, model)
        
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            
        if epoch % LOG_INTERVAL == 0 or best_val_acc == val_acc:
            log_message = f'epoch [{epoch}/{args.epochs}], loss: {loss:.3f}, train_acc: {train_acc:.3f}, val_acc: {val_acc:.3f}, test_acc: {test_acc:.3f}'
            print(log_message)
            
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
    # Settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=SUPPORTED_MODELS,
                        help=f'Choose one of the supported models: {", ".join(SUPPORTED_MODELS)}')
    parser.add_argument('--dataset', type=str, choices=SUPPORTED_DATASETS,
                        help=f'Choose one of the supported datasets: {", ".join(SUPPORTED_DATASETS)}')
    parser.add_argument('--split', type=str, choices=SUPPORTED_SPLITS, default='public',
                        help=f'Choose one of the supported splits: {", ".join(SUPPORTED_SPLITS)}')
    parser.add_argument('--n_trials', type=valid_positive_int, default=100, help='number of trials')
    parser.add_argument('--epochs', type=valid_positive_int, default=100, help='epochs per trial')
    args = parser.parse_args()
    
    
    dataset_path = os.path.join(DATA_DEFAULT_PATH, args.dataset)
    dataset = load_dataset.load_dataset(path=dataset_path,
                                       name=args.dataset,
                                       split=args.split,
                                       transform=T.TargetIndegree())
    
    x, edge_index, edge_attr = preprocess_data(dataset[0])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x, edge_index, edge_attr = x.to(device), edge_index.to(device), edge_attr.to(device)
    data = dataset[0].to(device)
        
    study_name = args.dataset + f'({args.split})' + '_' + args.model + '_study'
    storage_name = 'sqlite:///planetoid-study.db'

    study = optuna.create_study(storage=storage_name,
                                sampler=TPESampler(),
                                pruner=HyperbandPruner(),
                                study_name=study_name,                                
                                direction='maximize',
                                load_if_exists=True)
    study.optimize(lambda trial: objective(trial, data, dataset, args, device),
                   n_trials=args.n_trials)

    save_best_trial_to_json(study, args)
    display_results(study, args)