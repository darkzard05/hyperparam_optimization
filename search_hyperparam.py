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
import model

dataset_name = 'Cora' # dataset: Cora, PubMed, CiteSeer
model_name = 'appnp' # model: appnp, splineconv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device, 'is available')
dataset = Planetoid(root='/data/'+dataset_name,
                    name=dataset_name,
                    split='public',
                    transform=T.TargetIndegree()
                    )

data = dataset[0].to(device)

def define_model(trial):
    model_ = getattr(model, model_name)(dataset, trial)
    return model_.to(device)

def train(model, optimizer):
    model.train()
    optimizer.zero_grad()
    output, target = model(data)[data.train_mask], data.y[data.train_mask]
    loss = nn.CrossEntropyLoss()(output, target)
    loss.backward()
    optimizer.step()
    return loss

def test(model, mask):
    model.eval()
    with torch.no_grad():
        output = model(data)
        target = data.y
        pred = output.argmax(dim=1)
        correct = pred[mask] == target[mask]
        accuracy = int(correct.sum()) / int(mask.sum())
    return accuracy
        
def objective(trial):
    model = define_model(trial)
    
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True)
    eps = trial.suggest_float('eps', 1e-10, 1e-6, log=True)
    optim_name = trial.suggest_categorical('optimizer',
                                           ['Adam', 'NAdam', 'AdamW', 'RAdam'])
    
    optimizer = getattr(optim, optim_name)(model.parameters(),
                                           lr=lr,
                                           eps=eps,
                                           weight_decay=weight_decay)
    
    model.reset_parameters()
    
    best_val_acc = 0
    best_test_acc = 0
    
    for epoch in range(1, 201):
        loss = train(model, optimizer)
        train_acc = test(model, data.train_mask)
        val_acc = test(model, data.val_mask)
        test_acc = test(model, data.test_mask)
        
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            print(f'epoch: {epoch}, loss: {loss:.3f}, train_acc: {train_acc:.3f}, val_acc: {val_acc}, test_acc: {test_acc}')
            
        trial.report(best_test_acc, epoch)
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return best_test_acc

study_name = dataset_name + '_' + model_name + '_study'
storage_name = f'sqlite:///{study_name}.db'
n_trials = 50

study = optuna.create_study(study_name = study_name,
                            storage = storage_name,
                            load_if_exists= True,
                            sampler=TPESampler(),
                            pruner=HyperbandPruner(),
                            direction='maximize')
study.optimize(objective, n_trials=n_trials)

pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial
print("  Value: ", trial.value)
print("  Parameters: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
    
plot_param_importances(study).show()
plot_optimization_history(study).show()
plot_intermediate_values(study).show()
plot_slice(study).show()