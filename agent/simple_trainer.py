import torch
from agent.base_setup import BaseSetup 
from agent.base_trainer import BaseTrainer
from util.console_color import Prints
from torch.utils.data.dataloader import DataLoader
from  logger.metric_logger import MetricLogger


class SimpleTrainer(BaseTrainer):
    def __init__(self, setup: BaseSetup):
        super().__init__(setup)

    def _train_one_epoch(self, verbose=100):
        metric_logger = MetricLogger(
            ['Loss'],
            Prints.train,
            self.epoch,
            self.num_epoch,
            len(self.train_dataloader),
            Prints.info,
            verbose=verbose
        )
        self.model.train()

        for i, batch in enumerate(self.train_dataloader):
            input, target = batch
            input: torch.Tensor
            target: torch.Tensor
            input = input.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            
            output = self.model(input)

            loss = self.criterion(input=output, target=target)

            self.optimizer.zero_grad()
            if self.has_apex:
                from apex import amp
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            self.optimizer.step()

            metric_logger.log('Train', i, {'Loss': loss})
    
        metric_logger.global_avg('Train summary')    

    @torch.no_grad() 
    def _validate(self, dataloader: DataLoader, verbose=100):
        metric_logger = MetricLogger(
            ['Loss'],
            Prints.evaluate,
            self.epoch,
            self.num_epoch,
            len(dataloader),
            Prints.info,
            verbose=verbose
        )
        self.model.eval()
        for i, batch in enumerate(dataloader):
            input, target = batch
            input: torch.Tensor
            target: torch.Tensor
            input = input.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            
            output = self.model(input)
            loss = self.criterion(input=output, target=target)

            metric_logger.log('Valid', i, {'Loss': loss})

        metric_logger.global_avg('Valid summary')    

    @torch.no_grad()
    def test(self):
        self._validate(self.test_dataloader) 
    
    def train(self):
        for self.epoch in range(self.num_epoch):
            if self.distributed:
                self.train_dataloader.sampler.set_epoch(self.epoch)
            self._train_one_epoch()
            self._validate(self.valid_dataloader)