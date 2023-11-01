import os
import json
import datetime
import argparse
from functools import partial

import torch
import torch.nn as nn
from torch import optim
import torch_geometric.transforms as T

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from optuna.trial import TrialState

import nn_model
from load_dataset import get_dataset, get_train_loader
from settings import DATA_DEFAULT_PATH, LOG_INTERVAL
from hyperparams_utils import (get_common_model_params, add_extra_model_params,
                               get_optim_params)
from hyperparams_config import SUPPORTED_MODELS, SUPPORTED_DATASETS, SUPPORTED_SPLITS

def preprocess_data(data):
    data.x = (data.x - data.x.mean()) / data.x.std()
    return data


def train(data, model, optimizer, device):
    data = data.to(device)
    model.train()
    optimizer.zero_grad()
    output = model(data.x, data.edge_index, data.edge_attr)[data.train_mask]
    target = data.y[data.train_mask]
    loss = nn.CrossEntropyLoss()(output, target)
    loss.backward()
    optimizer.step()
    return loss


def test(data, model, mask, device):
    data = data.to(device)
    model.eval()
    with torch.no_grad():
        output = model(data.x, data.edge_index, data.edge_attr)
        target = data.y
        pred = output.argmax(dim=1)
        correct = pred[mask] == target[mask]
        accuracy = correct.sum() / mask.sum()
    return float(accuracy)


def evaluate(data, model, device):
    masks = [data.train_mask, data.val_mask, data.test_mask]
    results = [test(data, model, mask, device) for mask in masks]
    return tuple(results)


def initialize_model_and_optimizer(trial, dataset, args, device):
    # Initialize model
    basic_params = get_common_model_params(trial)
    extra_params = add_extra_model_params(trial, args.model, basic_params)
    model_params = {'in_channels': dataset[0].num_features,
                    'out_channels': dataset.num_classes,
                    **basic_params, **extra_params}
    model_class_name = args.model+'Model'
    model = getattr(nn_model, model_class_name)(**model_params).to(device)
    
    # Initialize optimizer
    optim_params = get_optim_params(trial)
    optimizer = getattr(optim, optim_params['optimizer'])(model.parameters(),
                                                          lr=optim_params['learning_rate'],
                                                          weight_decay=optim_params['weight_decay'])
    return model, optimizer


def objective(trial, train_loader, dataset, args, device):
    model, optimizer = initialize_model_and_optimizer(trial, dataset, args, device)
    
    best_val_acc, best_test_acc = 0, 0
    
    for epoch in range(1, args.epochs+1):
        total_loss, total_train_acc = 0, 0
        total_val_acc, total_test_acc = 0, 0
        
        for batch in train_loader:
            batch = preprocess_data(batch)
            loss = train(batch, model, optimizer, device)
            train_acc, val_acc, test_acc = evaluate(batch, model, device)
            total_loss += loss
            total_train_acc += train_acc
            total_val_acc += val_acc
            total_test_acc += test_acc
        loss = total_loss / len(train_loader)
        train_acc = total_train_acc / len(train_loader)
        val_acc = total_val_acc / len(train_loader)
        test_acc = total_test_acc / len(train_loader)
                
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
    
    filename = f'best_trial_{args.model}_{args.dataset}.json'
    
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
    Dataset name(split type): {args.dataset}
    Best trial:
      Value: {trial.value:.4f}
    Parameters:''')
    
    for key, value in trial.params.items():
        value_str = f'{value:.4f}' if isinstance(value, float) else str(value)
        print(f'      {key}: {value_str}')
    

def valid_positive_int(x):
    try:
        x_int = int(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f'{x} is not a integer.')
    
    if x_int <= 0:
        raise argparse.ArgumentTypeError(f'{x} is not a positive integer.')
    
    return x_int


def parser_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=SUPPORTED_MODELS,
                        help=f'Choose one of the supported models: {", ".join(SUPPORTED_MODELS)}')
    parser.add_argument('--dataset', type=str, choices=SUPPORTED_DATASETS,
                        help=f'Choose one of the supported datasets: {", ".join(SUPPORTED_DATASETS)}')
    parser.add_argument('--split', type=str, choices=SUPPORTED_SPLITS, default='public',
                        help=f'Choose one of the supported splits: {", ".join(SUPPORTED_SPLITS)}')
    parser.add_argument('--n_trials', type=valid_positive_int, default=100, help='number of trials')
    parser.add_argument('--epochs', type=valid_positive_int, default=100, help='epochs per trial')
    parser.add_argument('--batch_size', type=int, default=64, help='set data per iteration')
    parser.add_argument('--num_neighbors', type=list, default=[10, 20], help='neighbors sampled in graph layers')
    return parser.parse_args()


def main(args):
    dataset_path = os.path.join(DATA_DEFAULT_PATH, args.dataset)
    dataset = get_dataset(path=dataset_path, name=args.dataset, split=args.split,
                           transform=T.TargetIndegree())
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = dataset[0]
    
    train_loader = get_train_loader(data=data, num_neighbors=args.num_neighbors,
                                     batch_size=args.batch_size)
    
    study_name = args.dataset + '_' + args.model + '_study'
    storage_name = 'sqlite:///planetoid-study.db'

    study = optuna.create_study(storage=storage_name,
                                sampler=TPESampler(),
                                pruner=HyperbandPruner(),
                                study_name=study_name,                                
                                direction='maximize',
                                load_if_exists=True)
    partial_objective = partial(objective, train_loader=train_loader,
                                dataset=dataset, args=args, device=device)
    study.optimize(partial_objective, n_trials=args.n_trials)

    save_best_trial_to_json(study, args)
    display_results(study, args)


if __name__ == '__main__':
    args = parser_arguments()
    main(args)