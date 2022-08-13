from lib2to3.pgen2.token import OP
import torch.nn as nn
import torch
from ruamel import yaml
from munch import Munch
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer


class BaseSetup():
    train_dataloader: DataLoader
    valid_dataloader: DataLoader
    test_dataloader:  DataLoader
    model: nn.Module
    optimizer: Optimizer
    device: torch.device
    criterion = None
    has_apex: bool
    num_epoch: int
    distributed: bool


    def __init__(self, yaml_config_path) -> None:
        self._model
        self.config_dict = yaml.load(open(yaml_config_path, 'r'), Loader=yaml.Loader)
        self.config = Munch.fromDict(self.config_dict)

    def _random(self) -> None:
        raise NotImplementedError

    def _cuda(self) -> None:
        raise NotImplementedError

    def _dist(self) -> None:
        raise NotImplementedError

    def _logger(self) -> None:
        raise NotImplementedError

    def _data(self):
        raise NotImplementedError

    def _model(self):
        raise NotImplementedError

    def _optim(self):
        raise NotImplementedError

    def _lr_sched(self):
        raise NotImplementedError

    def _apex(self):
        raise NotImplementedError
    
    def _loss(self):
        raise NotImplementedError

    def setup_pipeline(self):
        raise NotImplementedError