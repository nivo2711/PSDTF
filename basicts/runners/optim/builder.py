

from typing import Dict

from torch import nn, optim
from torch.optim import lr_scheduler

from . import lr_schedulers as basicts_lr_scheduler
from . import optimizers as basicts_optim


def build_optim(optim_cfg: Dict, model: nn.Module) -> optim.Optimizer:

    if isinstance(optim_cfg['TYPE'], type):
        optim_type = optim_cfg['TYPE']
    else:
        if hasattr(optim, optim_cfg['TYPE']):
            optim_type = getattr(optim, optim_cfg['TYPE'])
        else:
            optim_type = getattr(basicts_optim, optim_cfg['TYPE'])
    optim_param = optim_cfg['PARAM'].copy()
    optimizer = optim_type(model.parameters(), **optim_param)
    return optimizer


def build_lr_scheduler(lr_scheduler_cfg: Dict, optimizer: optim.Optimizer) -> lr_scheduler._LRScheduler:

    lr_scheduler_cfg['TYPE'] = lr_scheduler_cfg['TYPE']
    if isinstance(lr_scheduler_cfg['TYPE'], type):
        scheduler_type = lr_scheduler_cfg['TYPE']
    else:
        if hasattr(lr_scheduler, lr_scheduler_cfg['TYPE']):
            scheduler_type = getattr(lr_scheduler, lr_scheduler_cfg['TYPE'])
        else:
            scheduler_type = getattr(basicts_lr_scheduler, lr_scheduler_cfg['TYPE'])
    scheduler_param = lr_scheduler_cfg['PARAM'].copy()
    scheduler_param['optimizer'] = optimizer
    scheduler = scheduler_type(**scheduler_param)
    return scheduler
