import torch.nn as nn


LOSS = {
    'CrossEntropyLoss': nn.CrossEntropyLoss,
    'MSELoss': nn.MSELoss,
    'NLLLoss': nn.NLLLoss,
}


def get_loss(loss_name, *args, **kwargs):
    return LOSS[loss_name](*args, **kwargs)