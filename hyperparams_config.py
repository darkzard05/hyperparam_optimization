SUPPORTED_MODELS = ['APPNP', 'Splineconv', 'GAT']
SUPPORTED_DATASETS = ['Cora', 'PubMed', 'CiteSeer', 'Reddit']
SUPPORTED_SPLITS = ['public', 'random', 'full', 'geom-gcn']

COMMON_MODEL_PARAMS = {
    'activation': ['relu', 'leakyrelu', 'I', 'elu', 'tanh', 'prelu'],
    'dropout': [0.0, 0.7],
    'n_units': [2** i for i in range(2, 8)],
    'num_layers': [1, 3]
}

EXTRA_MODEL_PARAMS = {
    'APPNP': {'K', 'alpha'},
    'Splineconv': {'kernel_size'},
    'GAT': {'heads'}
    }

EXTRA_PARAMS_TYPES = {
    'K': ('int', [5, 2e+2]),
    'alpha': ('float', [5e-2, 2e-1]),
    'kernel_size': ('int', [1, 8]),
    'heads': ('int', [1, 8])
}

OPTIM_PARAMS = {
    'optimizer': ['Adam', 'NAdam', 'AdamW', 'RAdam'],
    'learning_rate': [1e-3, 1e-1],
    'weight_decay': [1e-5, 1e-1],
}