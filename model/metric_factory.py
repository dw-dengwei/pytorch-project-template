import torch.nn as nn
import torch


def accuracy(input, target, output_predict=False):
    with torch.no_grad():
        pred = torch.argmax(input, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()

    if output_predict:
        return correct / len(target), pred
    else:
        return correct / len(target)

METRIC = {
    'MSE': nn.MSELoss,
    'ACC': accuracy,
}


def get_loss(metric_name, *args, **kwargs):
    return METRIC[metric_name](*args, **kwargs)
