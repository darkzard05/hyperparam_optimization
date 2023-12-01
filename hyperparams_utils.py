from hyperparams_config import (COMMON_MODEL_PARAMS, EXTRA_MODEL_PARAMS,
                                EXTRA_PARAMS_TYPES, OPTIM_PARAMS)

def get_common_model_params(trial):
    model_params = {
        'activation': trial.suggest_categorical('activation', COMMON_MODEL_PARAMS['activation']),
        'dropout': trial.suggest_float('dropout', *COMMON_MODEL_PARAMS['dropout']),
        'num_layers': trial.suggest_int('num_layers', *COMMON_MODEL_PARAMS['num_layers']),
        }
    model_params.update({'n_units': [trial.suggest_categorical(f'n_units_{i+1}', COMMON_MODEL_PARAMS['n_units'])
                                     for i in range(model_params['num_layers']-1)] if model_params['num_layers'] > 1 else None})
    return model_params


def add_extra_model_params(trial, model_name):
    params = dict()
    for param in EXTRA_MODEL_PARAMS[model_name]:
        param_type, param_range = EXTRA_PARAMS_TYPES[param]
        if param_type == 'categorical':
            suggest = trial.suggest_categorical(param, param_range)
        elif param_type == 'int':
            suggest = trial.suggest_int(param, *param_range)
        elif param_type == 'float':
            suggest = trial.suggest_float(param, *param_range)
        params[param] = suggest
    return params


def get_optim_params(trial):
    optim_params = {
        'learning_rate': trial.suggest_float('learning_rate', *OPTIM_PARAMS['learning_rate'], log=True),
        'weight_decay': trial.suggest_float('weight_decay', *OPTIM_PARAMS['weight_decay'], log=True),
        'optimizer': trial.suggest_categorical('optimizer', OPTIM_PARAMS['optimizer']),
        }
    return optim_params