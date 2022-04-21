from optim.loss import *


def get_optim(model: torch.nn.Module, config:dict):
    assert 'name' in config.keys()
    if config['name'] == 'sgd':
        return get_sgd(model, config)
    elif config['name'] == 'adam':
        return get_adam(model, config)
