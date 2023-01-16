import argparse

import torch
import torch.nn as nn
from torch import optim
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

import optuna
from optuna.visualization import *
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from optuna.trial import TrialState

import nn_model

def train(data, model, optimizer):
    model.train()
    optimizer.zero_grad()
    output, target = model(data)[data.train_mask], data.y[data.train_mask]
    loss = nn.CrossEntropyLoss()(output, target)
    loss.backward()
    optimizer.step()
    return loss

def test(data, model, mask):
    model.eval()
    with torch.no_grad():
        output = model(data)
        target = data.y
        pred = output.argmax(dim=1)
        correct = pred[mask] == target[mask]
        accuracy = int(correct.sum()) / int(mask.sum())
    return accuracy
        
def objective(trial):
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True)
    eps = trial.suggest_float('eps', 1e-10, 1e-6, log=True)
    # optim_name = trial.suggest_categorical('optimizer',
    #                                        ['Adam', 'NAdam', 'AdamW', 'RAdam'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = getattr(nn_model, args.model)(dataset, trial)
    model.to(device)
    
    # optimizer = getattr(optim, optim_name)(model.parameters(),
    #                                        lr=lr,
    #                                        eps=eps,
    #                                        weight_decay=weight_decay)
    optimizer = optim.Adam(model.parameters(),
                           lr=lr,
                           eps=eps,
                           weight_decay=weight_decay)
    model.reset_parameters()
    
    best_val_acc = 0
    best_test_acc = 0
    
    for epoch in range(1, args.epochs+1):
        loss = train(data, model, optimizer)
        train_acc = test(data, model, data.train_mask)
        val_acc = test(data, model, data.val_mask)
        test_acc = test(data, model, data.test_mask)
        
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            print(f'epoch: {epoch}, loss: {loss:.3f}, train_acc: {train_acc:.3f}, val_acc: {val_acc}, test_acc: {test_acc}')
            
        trial.report(val_acc, epoch)
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return best_test_acc
    
if __name__ == '__main__':
    # settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trials', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--dataset', type=str) # dataset: Cora, PubMed, CiteSeer
    parser.add_argument('--model', type=str) # model: appnp, splineconv, gat
    parser.add_argument('--split', type=str, default='public') # dataset split: public, random, full, geom-gcn
    args = parser.parse_args()
    
    torch.cuda.empty_cache()
    
    dataset = Planetoid(root='/data/'+args.dataset,
                        name=args.dataset,
                        split=args.split,
                        transform=T.TargetIndegree()
                        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = dataset[0].to(device)
    
    study_name = args.dataset + f'({args.split})' + '_' + args.model + '_study'
    storage_name = f'sqlite:///{study_name}.db'

    study = optuna.create_study(study_name = study_name,
                                storage = storage_name,
                                load_if_exists= True,
                                sampler=TPESampler(),
                                pruner=HyperbandPruner(),
                                direction='maximize')
    study.optimize(objective, n_trials=args.n_trials)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print('Study statistics: ')
    print('  Number of finished trials: ', len(study.trials))
    print('  Number of pruned trials: ', len(pruned_trials))
    print('  Number of complete trials: ', len(complete_trials))

    print(f'Model name: {args.model}')
    print(f'Dataset name: {args.dataset}({args.split})')
    print('Best trial:')
    trial = study.best_trial
    print('  Value: ', trial.value)
    print('  Parameters: ')
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    plot_param_importances(study).show()
    plot_optimization_history(study).show()
    plot_intermediate_values(study).show()
    plot_parallel_coordinate(study).show()
    plot_contour(study).show()
    plot_slice(study).show()