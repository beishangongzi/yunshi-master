import torch

from util import get_map_val


# SGD
def get_sgd(model: torch.nn.Module, config):
    lr = get_map_val(config, 'lr', 1e-2)
    momentum = get_map_val(config, 'momentum', 0)
    weight_decay = get_map_val(config, 'weight_decay', 0)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    return optimizer


# adam
def get_adam(model: torch.nn.Module, config):
    lr = get_map_val(config, 'lr', 1e-2)
    momentum = get_map_val(config, 'momentum', 0)
    weight_decay = get_map_val(config, 'weight_decay', 0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=[momentum, 0.999], weight_decay=weight_decay)
    return optimizer