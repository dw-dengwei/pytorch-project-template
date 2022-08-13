import torch
import random
import os
import wandb
import time
import util.distributed
import util.data
import numpy as np
import torch.backends.cudnn as cudnn
from util.console_color import Prints
from agent.base_setup import BaseSetup
from optim.optim_factory import get_optim
from lr_sched.sched_factory  import get_sched
from model.model_factory import get_model
from model.loss_factory import get_loss


class SimpleSetup(BaseSetup):
    def __init__(self, yaml_config_path) -> None:
        super().__init__(yaml_config_path)
        # environmental settings
        self.seed = self.config.random_seed
        self.device = torch.device(self.config.device)
        self.dist_url = self.config.dist_url
        self.project_name = self.config.project_name
        self.run_name = '' if self.config.run_name is None else ' ' + self.config.run_name
        self.dataset_name = self.config.dataset_name
        self.opt_level = self.config.opt_level
        self.mode = self.config.mode
        self.use_wandb = self.config.use_wandb
        self.loss_name = self.config.loss_name

        # training hyperparameters
        self.train_batch_size = self.config.train_batch_size
        self.valid_batch_size = self.config.valid_batch_size
        self.test_batch_size = self.config.test_batch_size
        self.num_epoch = self.config.num_epoch

        # model hyperparameters
        self.model_name = self.config.model_name
        self.in_features = self.config.in_features
        self.out_features = self.config.out_features

        # lr_scheduler & optimizer
        self.optim_config = self.config.optim
        self.sched_config = getattr(self.config, 'sched', None)

    def setup_pipeline(self):
        funs = [
            self._random,
            self._cuda,
            self._dist,
            self._logger,
            self._data,
            self._model,
            self._optim,
            self._lr_sched,
            self._apex,
            self._loss,
        ]
        for f in funs:
            f() 

    def _random(self) -> None:
        Prints.heading(f'Setting random seed: {self.seed}')
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

    def _cuda(self) -> None:
        if self.config.device == 'cpu':
            self.has_cuda = False 
            Prints.warning('NOT using cuda')
        else:
            cudnn.benchmark = True 
            self.has_cuda = True
            Prints.ok('Using cuda')

    def _only_master_print(self, is_master: bool):
        """
        This function disables printing when not in master process
        """
        import builtins as __builtin__
        builtin_print = __builtin__.print

        def print(*args, **kwargs):
            force = kwargs.pop('force', False)
            if is_master or force:
                builtin_print(*args, **kwargs)

        __builtin__.print = print

    def _dist(self) -> None:
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            self.rank = int(os.environ["RANK"])
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.gpu = int(os.environ['LOCAL_RANK'])
            Prints.ok('Using distributed mode')
            self.distributed = True
        elif 'SLURM_PROCID' in os.environ:
            self.rank = int(os.environ['SLURM_PROCID'])
            self.gpu = self.rank % torch.cuda.device_count()
            Prints.ok('Using distributed mode')
            self.distributed = True
        else:
            Prints.warning('NOT using distributed mode')
            self.distributed = False
            return

        torch.cuda.set_device(self.gpu)
        self.dist_backend = 'nccl'
        Prints.info(f'Distributed init (rank {self.rank}): {self.dist_url}', flush=True)
        torch.distributed.init_process_group(
            backend=self.dist_backend,
            init_method=self.dist_url,
            world_size=self.world_size,
            rank=self.rank
        )
        torch.distributed.barrier()
        self._only_master_print(self.rank == 0)

    def _logger(self) -> None:
        if self.use_wandb:
            Prints.ok('Using wandb')
            wandb.init(
                project=self.project_name,
                name=time.asctime() + self.run_name,
                config=self.config_dict
            )
        else:
            Prints.warning('NOT using wandb')

    def _data(self):
        Prints.heading('Creating dataloader')
        train_valid_test_datasets = util.data.create_dataset(self.dataset_name, self.config) 

        if self.distributed:
            num_tasks = util.distributed.get_world_size()
            global_rank = util.distributed.get_rank()
            samplers = util.data.create_sampler(
                train_valid_test_datasets, 
                [True, False, False], 
                num_tasks, 
                global_rank
            )         
        else:
            samplers = [None, None, None]

        self.train_dataloader, self.valid_dataloader, self.test_dataloader = \
            util.data.create_loader(
                train_valid_test_datasets,
                samplers,
                batch_size=[self.train_batch_size] +
                           [self.valid_batch_size] +
                           [self.test_batch_size],
                num_workers=[0] * 3,
                is_trains=[True,False,False], 
                collate_fns=[None] * 3
        )

    def _load_pretrain(self):
        return
        Prints.heading('Loading checkpoints')

    def _model(self):
        Prints.heading('Creating model')
        model = get_model(
            self.model_name, 
            **{
                'in_features': self.in_features, 
                'out_features': self.out_features
            }
        )
        self._load_pretrain()
        model = model.to(self.device)
        if self.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model, 
                device_ids=[self.gpu]
            )

        self.model = model

    def _optim(self):
        Prints.heading('Initializing optimizer')
        self.optimizer = get_optim(self.optim_config, self.model)

    def _lr_sched(self):
        Prints.heading('Initializing learning rate scheduler')
        if self._lr_sched is not None:
            self._lr_sched = get_sched(self.sched_config, self.optimizer)
            self.has_lr_sched = True
            Prints.ok('Using learning rate scheduler')
        else:
            self.has_lr_sched = False
            Prints.warning('NOT using learning rate scheduler')

    def _apex(self):
        try:
            import apex
            from apex import amp

            Prints.ok('Using apex')
            self.has_apex = True
            if self.distributed:
                self.model = apex.parallel.DistributedDataParallel(self.model)

            self.model, self.optimizer = amp.initialize(
                self.model, self.optimizer, opt_level=self.opt_level
            )

        except ImportError:
            self.has_apex = False
            Prints.warning('NOT using apex')
            return

    def _loss(self):
        self.criterion = get_loss(self.loss_name)