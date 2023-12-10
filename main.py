import os
import json
import datetime, time
import argparse
from tqdm import tqdm
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
from graph_dataloader import get_dataset, get_dataloader, preprocess_data
from settings import DATA_DEFAULT_PATH, LOG_INTERVAL
from hyperparams_utils import (get_common_model_params, add_extra_model_params,
                               get_optim_params)
from hyperparams_config import SUPPORTED_MODELS, SUPPORTED_DATASETS


def train_epoch(model, train_loader, optimizer, device) -> float:
    model.train()
    total_loss = 0
    loss_fn = nn.CrossEntropyLoss()
    
    for batch in tqdm(train_loader, desc='training'):
        x, edge_index, edge_attr = tuple(t.to(device) for t in preprocess_data(batch))
        optimizer.zero_grad()
        output = model(x, edge_index, edge_attr)[batch.train_mask]
        target = batch.y.to(device)[batch.train_mask]
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def evaluate(model, loader, device, mode='validation') -> float:
    model.eval()
    total_acc = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=mode):
            x, edge_index, edge_attr = tuple(t.to(device) for t in preprocess_data(batch))
            output = model(x, edge_index, edge_attr)
            target = batch.y.to(device)
            pred = output.argmax(dim=1)
            correct = (pred == target)
            accuracy = correct.sum().item() / len(target)
            total_acc += accuracy
            
    return total_acc / len(loader)


def initialize_model(trial, dataset,
                     args: argparse.Namespace):
    model_class_name = args.model+'Model'
    model_common_params = get_common_model_params(trial)
    
    model_params = {'in_channels': dataset[0].num_features,
                    'out_channels': dataset.num_classes,
                    **model_common_params}
    model_extra_params = add_extra_model_params(trial, args.model)
    model_params.update(model_extra_params)
    
    model_class = getattr(nn_model, model_class_name)
    model = model_class(model_params)
    return model


def initialize_optimizer(trial, model):
    optim_params = get_optim_params(trial)
    optimizer = optim.__dict__[optim_params['optimizer']](model.parameters(),
                                                          lr=optim_params['learning_rate'],
                                                          weight_decay=optim_params['weight_decay'])
    return optimizer


def objective(trial,
              train_loader, val_loader, test_loader, dataset,
              args: argparse.Namespace, device: torch.device) -> float:
    model = initialize_model(trial, dataset, args)
    optimizer = initialize_optimizer(trial, model)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1,
                                          gamma=0.85)
    
    best_val_acc, best_test_acc = 0, 0
    
    for epoch in range(1, args.epochs+1):
        model.to(device)
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_acc = evaluate(model, val_loader, device, mode='validation')
        test_acc = evaluate(model, test_loader, device, mode='testing')
        
        scheduler.step()
        
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            
        if epoch % LOG_INTERVAL == 0 or best_val_acc == val_acc:
            log_message = f'epoch [{epoch}/{args.epochs}], loss: {train_loss:.3f}, val_acc: {val_acc:.3f}, test_acc: {test_acc:.3f}'
            print(log_message)
            
        trial.report(val_acc, epoch)
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
    return best_test_acc


def save_best_trial_to_json(study,
                            args: argparse.Namespace):
    best_trial = study.best_trial
    
    result = {
        'date_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_name': args.model,
        'dataset_name': args.dataset,
        'best_trials': {
            'params': best_trial.params,
            'value': best_trial.value,
            'number': best_trial.number
            }
        }
    
    filename = f'best_trial_{args.model}_{args.dataset}.json'
    
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
    Dataset name: {args.dataset}
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
    parser.add_argument('--dataset', type=str, choices=SUPPORTED_DATASETS,
                        help=f'Choose one of the supported datasets: {", ".join(SUPPORTED_DATASETS)}')
    parser.add_argument('--model', type=str, choices=SUPPORTED_MODELS,
                        help=f'Choose one of the supported models: {", ".join(SUPPORTED_MODELS)}')
    parser.add_argument('--n_trials', type=valid_positive_int, help='number of trials')
    parser.add_argument('--epochs', type=valid_positive_int, help='epochs per trial')
    parser.add_argument('--batch_size', type=int, help='set data per iteration')
    parser.add_argument('--num_neighbors', type=eval, help='neighbors sampled in graph layers')
    parser.add_argument('--num_workers', type=int, default=0, help='adjust the workers for fast data loading')
    
    return parser.parse_args()


def main(args: argparse.Namespace):
    dataset_path = os.path.join(DATA_DEFAULT_PATH, args.dataset)
    dataset = get_dataset(path=dataset_path, name=args.dataset, transform=T.TargetIndegree())
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = dataset[0]
    print(f'''dataset name: {args.dataset}
    number of nodes: {data.num_nodes}
    number of features: {data.num_node_features}
    number of train nodes: {data.train_mask.sum()}
    number of val nodes: {data.val_mask.sum()}
    number of test nodes: {data.test_mask.sum()}''')
    
    train_loader, val_loader, test_loader = get_dataloader(data=data,
                                                           num_neighbors=args.num_neighbors,
                                                           batch_size=args.batch_size,
                                                           num_workers=args.num_workers)

    study_name = f'{args.dataset}_{args.model}_study'
    storage_name = 'sqlite:///planetoid-study.db'

    study = optuna.create_study(storage=storage_name,
                                sampler=TPESampler(consider_prior=True,
                                                   n_startup_trials=5,
                                                   multivariate=False),
                                pruner=HyperbandPruner(),
                                study_name=study_name,                                
                                direction='maximize',
                                load_if_exists=True)
   
    partial_objective = partial(objective,
                                train_loader=train_loader,
                                val_loader=val_loader,
                                test_loader=test_loader,
                                dataset=dataset,
                                args=args,
                                device=device)
            
    study.optimize(partial_objective, n_trials=args.n_trials)

    save_best_trial_to_json(study, args)
    display_results(study, args)


if __name__ == '__main__':
    args = parser_arguments()
    main(args)