import torch
from agent.base_setup import BaseSetup


class BaseTrainer():
    def __init__(self, setup: BaseSetup):
        self.distributed = setup.distributed
        self.device = setup.device
        self.config = setup.config
        self.has_apex = setup.has_apex

        self.num_epoch = setup.num_epoch
        self.train_dataloader = setup.train_dataloader
        self.valid_dataloader = setup.valid_dataloader
        self.test_dataloader = setup.test_dataloader

        self.model = setup.model
        self.optimizer = setup.optimizer
        self.criterion = setup.criterion
    
    def _train_one_epoch(self):
        raise NotImplementedError
    
    @torch.no_grad()
    def _validate(self):
        raise NotImplementedError
    
    def train(self):
        raise NotImplementedError

    def final(self):
        raise NotImplementedError