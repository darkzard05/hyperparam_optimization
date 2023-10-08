DATA_DEFAULT_PATH = '/data'
LOG_INTERVAL = 10

SUPPORTED_MODELS = ['APPNP', 'Splineconv', 'GAT']
SUPPORTED_DATASETS = ['Cora', 'PubMed', 'CiteSeer', 'Reddit']
SUPPORTED_SPLITS = ['public', 'random', 'full', 'geom-gcn']

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