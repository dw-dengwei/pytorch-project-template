""" Scheduler Factory
Hacked together by / Copyright 2020 Ross Wightman
"""
from .cosine import CosineLRScheduler
from .tanh import TanhLRScheduler
from .step import StepLRScheduler
from .plateau import PlateauLRScheduler


def get_sched(sched_config, optimizer):
    num_epochs = sched_config.epochs

    if getattr(sched_config, 'lr_noise', None) is not None:
        lr_noise = getattr(sched_config, 'lr_noise')
        if isinstance(lr_noise, (list, tuple)):
            noise_range = [n * num_epochs for n in lr_noise]
            if len(noise_range) == 1:
                noise_range = noise_range[0]
        else:
            noise_range = lr_noise * num_epochs
    else:
        noise_range = None

    lr_scheduler = None
    if sched_config.sched == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_epochs,
            t_mul=getattr(sched_config, 'lr_cycle_mul', 1.),
            lr_min=sched_config.min_lr,
            decay_rate=sched_config.decay_rate,
            warmup_lr_init=sched_config.warmup_lr,
            warmup_t=sched_config.warmup_epochs,
            cycle_limit=getattr(sched_config, 'lr_cycle_limit', 1),
            t_in_epochs=True,
            noise_range_t=noise_range,
            noise_pct=getattr(sched_config, 'lr_noise_pct', 0.67),
            noise_std=getattr(sched_config, 'lr_noise_std', 1.),
            noise_seed=getattr(sched_config, 'seed', 42),
        )
        num_epochs = lr_scheduler.get_cycle_length() + sched_config.cooldown_epochs
    elif sched_config.sched == 'tanh':
        lr_scheduler = TanhLRScheduler(
            optimizer,
            t_initial=num_epochs,
            t_mul=getattr(sched_config, 'lr_cycle_mul', 1.),
            lr_min=sched_config.min_lr,
            warmup_lr_init=sched_config.warmup_lr,
            warmup_t=sched_config.warmup_epochs,
            cycle_limit=getattr(sched_config, 'lr_cycle_limit', 1),
            t_in_epochs=True,
            noise_range_t=noise_range,
            noise_pct=getattr(sched_config, 'lr_noise_pct', 0.67),
            noise_std=getattr(sched_config, 'lr_noise_std', 1.),
            noise_seed=getattr(sched_config, 'seed', 42),
        )
        num_epochs = lr_scheduler.get_cycle_length() + sched_config.cooldown_epochs
    elif sched_config.sched == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=sched_config.decay_epochs,
            decay_rate=sched_config.decay_rate,
            warmup_lr_init=sched_config.warmup_lr,
            warmup_t=sched_config.warmup_epochs,
            noise_range_t=noise_range,
            noise_pct=getattr(sched_config, 'lr_noise_pct', 0.67),
            noise_std=getattr(sched_config, 'lr_noise_std', 1.),
            noise_seed=getattr(sched_config, 'seed', 42),
        )
    elif sched_config.sched == 'plateau':
        mode = 'min' if 'loss' in getattr(sched_config, 'eval_metric', '') else 'max'
        lr_scheduler = PlateauLRScheduler(
            optimizer,
            decay_rate=sched_config.decay_rate,
            patience_t=sched_config.patience_epochs,
            lr_min=sched_config.min_lr,
            mode=mode,
            warmup_lr_init=sched_config.warmup_lr,
            warmup_t=sched_config.warmup_epochs,
            cooldown_t=0,
            noise_range_t=noise_range,
            noise_pct=getattr(sched_config, 'lr_noise_pct', 0.67),
            noise_std=getattr(sched_config, 'lr_noise_std', 1.),
            noise_seed=getattr(sched_config, 'seed', 42),
        )

    return lr_scheduler, num_epochs
