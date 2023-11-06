import os
import json
import datetime
import argparse
from typing import Tuple
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


def preprocess_data(data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = (data.x - data.x.mean()) / data.x.std()
    edge_index = data.edge_index
    edge_attr = data.edge_attr
    return x, edge_index, edge_attr


def train(x: torch.Tensor,
          edge_index: torch.Tensor,
          edge_attr: torch.Tensor,
          data, model, optimizer, device) -> torch.Tensor:
    model.train()
    optimizer.zero_grad()
    output = model(x.to(device), edge_index.to(device), edge_attr.to(device))[data.train_mask]
    target = data.y.to(device)[data.train_mask]
    loss = nn.CrossEntropyLoss()(output, target)
    loss.backward()
    optimizer.step()
    return loss


def test(x: torch.Tensor,
         edge_index: torch.Tensor,
         edge_attr: torch.Tensor,
         data, model, mask, device) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        output = model(x.to(device), edge_index.to(device), edge_attr.to(device))
        target = data.y.to(device)
        pred = output.argmax(dim=1)
        correct = pred[mask] == target[mask]
        accuracy = correct.sum() / mask.sum()
    return accuracy


def evaluate(x: torch.Tensor,
             edge_index: torch.Tensor,
             edge_attr: torch.Tensor,
             data, model, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    masks = [data.train_mask, data.val_mask, data.test_mask]
    results = [test(x, edge_index, edge_attr, data, model, mask, device) for mask in masks]
    return tuple(results)


def initialize_model_and_optimizer(trial, dataset,
                                   args: argparse.Namespace,
                                   device: torch.device):
    # Initialize model
    model_basic_params = get_common_model_params(trial)
    model_extra_params = add_extra_model_params(trial, args.model, model_basic_params)
    model_params = {'in_channels': dataset[0].num_features,
                    'out_channels': dataset.num_classes,
                    **model_basic_params, **model_extra_params}
    model_class_name = args.model+'Model'
    model = getattr(nn_model, model_class_name)(**model_params)
    model.to(device)
    
    # Initialize optimizer
    optim_params = get_optim_params(trial)
    optimizer = getattr(optim, optim_params['optimizer'])(model.parameters(),
                                                          lr=optim_params['learning_rate'],
                                                          weight_decay=optim_params['weight_decay'])
    return model, optimizer


def objective(trial, train_loader, data, dataset,
              x: torch.Tensor,
              edge_index: torch.Tensor,
              edge_attr: torch.Tensor,
              args: argparse.Namespace,
              device: torch.device) -> torch.Tensor:
    model, optimizer = initialize_model_and_optimizer(trial, dataset, args, device)
    
    best_val_acc, best_test_acc = 0, 0
    
    early_stopping_counter = 0
    early_stopping_patience = 10
    
    for epoch in range(1, args.epochs+1):
        if args.dataset_type == 'Reddit':
            total_loss, total_train_acc = 0, 0
            total_val_acc, total_test_acc = 0, 0
            
            for batch in train_loader:
                x, edge_index, edge_attr = preprocess_data(batch)
                x, edge_index, edge_attr = x.to(device), edge_index.to(device), edge_attr.to(device)
                loss = train(x, edge_index, edge_attr, batch, model, optimizer, device)
                train_acc, val_acc, test_acc = evaluate(x, edge_index, edge_attr, batch, model, device)
                total_loss += loss
                total_train_acc += train_acc
                total_val_acc += val_acc
                total_test_acc += test_acc
                            
            loss = total_loss / len(train_loader)
            train_acc = total_train_acc / len(train_loader)
            val_acc = total_val_acc / len(train_loader)
            test_acc = total_test_acc / len(train_loader)
        else:
            loss = train(x, edge_index, edge_attr, data, model, optimizer)
            train_acc, val_acc, test_acc = evaluate(x, edge_index, edge_attr, data, model)
        
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            
        if epoch % LOG_INTERVAL == 0 or best_val_acc == val_acc:
            log_message = f'epoch [{epoch}/{args.epochs}], loss: {loss:.3f}, train_acc: {train_acc:.3f}, val_acc: {val_acc:.3f}, test_acc: {test_acc:.3f}'
            print(log_message)
            
        trial.report(val_acc, epoch)
        
        if trial.should_prune() or early_stopping_counter >= early_stopping_patience:
            raise optuna.exceptions.TrialPruned()
        
    return best_test_acc


def save_best_trial_to_json(study,
                            args: argparse.Namespace):
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


def display_results(study,
                    args: argparse.Namespace):
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
    

def valid_positive_int(x) -> int:
    try:
        x_int = int(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f'{x} is not a integer.')
    
    if x_int <= 0:
        raise argparse.ArgumentTypeError(f'{x} is not a positive integer.')
    
    return x_int


def parser_arguments():
    parser = argparse.ArgumentParser()
    
    subparsers = parser.add_subparsers(dest='dataset_type', title='Reddit or Planetoid')
    
    parser_reddit = subparsers.add_parser('Reddit', description='Reddit')
    parser_planetoid = subparsers.add_parser('Planetoid', description='Planetoid')
    
    for p in [parser_reddit, parser_planetoid]:
        p.add_argument('--model', type=str, choices=SUPPORTED_MODELS,
                            help=f'Choose one of the supported models: {", ".join(SUPPORTED_MODELS)}')
        p.add_argument('--n_trials', type=valid_positive_int, default=100, help='number of trials')
        p.add_argument('--epochs', type=valid_positive_int, default=100, help='epochs per trial')
    
    parser_planetoid.add_argument('--dataset', type=str, choices=SUPPORTED_DATASETS,
                                  help=f'Choose one of the supported datasets: {", ".join(SUPPORTED_DATASETS)}')
    # parser_planetoid.add_argument('--split', type=str, choices=SUPPORTED_SPLITS, default='public',
    #                               help=f'Choose one of the supported splits: {", ".join(SUPPORTED_SPLITS)}')
    
    parser_reddit.add_argument('--batch_size', type=int, default=64, help='set data per iteration')
    parser_reddit.add_argument('--num_neighbors', type=list, default=[10, 20], help='neighbors sampled in graph layers')
    return parser.parse_args()


def main(args: argparse.Namespace):
    dataset_path = os.path.join(DATA_DEFAULT_PATH, args.dataset_type if args.dataset_type == 'Reddit'
                                else args.dataset)
    dataset = get_dataset(path=dataset_path, name=args.dataset_type if args.dataset_type == 'Reddit'
                          else args.dataset, transform=T.TargetIndegree())
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = dataset[0]
    x, edge_index, edge_attr = preprocess_data(data)

    if args.dataset_type == 'Reddit':
        train_loader = get_train_loader(data=data, num_neighbors=args.num_neighbors,
                                        batch_size=args.batch_size)

    study_name = args.dataset_type if args.dataset_type == 'Reddit' else args.dataset + '_' + args.model + '_study'
    storage_name = 'sqlite:///planetoid-study.db'

    study = optuna.create_study(storage=storage_name,
                                sampler=TPESampler(),
                                pruner=HyperbandPruner(),
                                study_name=study_name,                                
                                direction='maximize',
                                load_if_exists=True)
   
    partial_objective = partial(objective, train_loader=train_loader,
                                data=data, dataset=dataset,
                                x=x, edge_index=edge_index, edge_attr=edge_attr,
                                args=args, device=device)
    study.optimize(partial_objective, n_trials=args.n_trials)

    save_best_trial_to_json(study, args)
    display_results(study, args)


if __name__ == '__main__':
    args = parser_arguments()
    main(args)