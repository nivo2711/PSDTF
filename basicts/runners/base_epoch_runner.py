import logging
import os
import time
from abc import ABCMeta, abstractmethod
from typing import Dict, Optional, Tuple, Union

import setproctitle
import torch
from easytorch.config import get_ckpt_save_dir
from easytorch.core.checkpoint import (backup_last_ckpt, clear_ckpt, load_ckpt,
                                       save_ckpt)
from easytorch.core.data_loader import build_data_loader, build_data_loader_ddp
from easytorch.core.meter_pool import MeterPool
from easytorch.device import to_device
from easytorch.utils import (TimePredictor, get_local_rank, get_logger,
                             is_master, master_only, set_env)
from packaging import version
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..utils import get_dataset_name
from . import optim

class BaseEpochRunner(metaclass=ABCMeta):

    def __init__(self, cfg: Dict) -> None:

        self.logger = get_logger('easytorch')

        set_env(cfg.get('ENV', {}))
        self.to_running_device = to_device

        self.model_name = cfg['MODEL.NAME']
        self.ckpt_save_dir = get_ckpt_save_dir(cfg)
        self.logger.info('Set ckpt save dir: \'{}\''.format(self.ckpt_save_dir))
        self.ckpt_save_strategy = None
        self.num_epochs = None
        self.start_epoch = None
        self.total_iters = None
        self.iter_per_epoch = None

        self.val_interval = cfg.get('VAL', {}).get('INTERVAL', 1)
        self.test_interval = cfg.get('TEST', {}).get('INTERVAL', 1)

        if not os.path.isdir(self.ckpt_save_dir):
            os.makedirs(self.ckpt_save_dir)

        self.model = self.build_model(cfg)

        self.optim = None
        self.scheduler = None

        self.train_data_loader = None
        self.val_data_loader = None
        self.test_data_loader = None

        self.meter_pool = None

        self.tensorboard_writer = None

        self.kl_loss = nn.KLDivLoss(reduction='none')

        self.early_stopping_patience = cfg.get('TRAIN', {}).get('EARLY_STOPPING_PATIENCE', None)
        self.current_patience = self.early_stopping_patience
        assert self.early_stopping_patience is None or self.early_stopping_patience > 0, 'Early stopping patience must be a positive integer.'

        proctitle_name = f"{cfg['MODEL'].get('NAME')}({get_dataset_name(cfg)})"
        setproctitle.setproctitle(f'{proctitle_name}@BasicTS')

    def define_model(self, cfg: Dict) -> nn.Module:

        return cfg['MODEL']['ARCH'](**cfg['MODEL']['PARAM'])

    def build_model(self, cfg: Dict) -> nn.Module:

        self.logger.info('Building model.')
        model = self.define_model(cfg)
        model = self.to_running_device(model)

        if cfg.get('TRAIN.COMPILE_MODEL', False):
            current_version = torch.__version__
            if version.parse(current_version) >= version.parse('2.0'):
                self.logger.info('Compile model with torch.compile')
                model = torch.compile(model)
            else:
                self.logger.warning(f'torch.compile requires PyTorch 2.0 or higher. Current version: {current_version}. Skipping compilation.')

        if torch.distributed.is_initialized():
            print('torch.distributed.is_initialized')
            model = DDP(
                model,
                device_ids=[get_local_rank()],
                find_unused_parameters=cfg.get('MODEL.DDP_FIND_UNUSED_PARAMETERS', False)
            )
        return model

    def build_optim(self, optim_cfg: Dict, model) -> optim.Optimizer:

        return optim.build_optim(optim_cfg, model)

    def build_lr_scheduler(self, cfg: Dict) -> None:
        if cfg.has('TRAIN.LR_SCHEDULER'):
            self.scheduler = optim.build_lr_scheduler(cfg['TRAIN.LR_SCHEDULER'], self.optim)
            self.logger.info('Set lr_scheduler: {}'.format(self.scheduler))
            self.register_epoch_meter('train/lr', 'train', '{:.2e}')

    def build_train_data_loader(self, cfg: Dict) -> DataLoader:

        self.logger.info('Building training data loader.')
        dataset, self.iter_per_epoch = self.build_train_dataset(cfg)
        if torch.distributed.is_initialized():
            return build_data_loader_ddp(dataset, cfg['TRAIN.DATA'])
        else:
            return build_data_loader(dataset, cfg['TRAIN.DATA'])

    @abstractmethod
    def build_train_dataset(self, cfg: Dict) -> Dataset:

        pass

    def build_val_data_loader(self, cfg: Dict) -> DataLoader:

        self.logger.info('Building val data loader.')
        dataset = self.build_val_dataset(cfg)
        return build_data_loader(dataset, cfg['VAL.DATA'])

    @staticmethod
    def build_val_dataset(cfg: Dict) -> Dataset:

        raise NotImplementedError()

    def build_test_data_loader(self, cfg: Dict) -> DataLoader:

        dataset = self.build_test_dataset(cfg)
        return build_data_loader(dataset, cfg['TEST']['DATA'])

    def build_test_dataset(self, cfg: Dict) -> Dataset:

        raise NotImplementedError('build_test_dataset method must be implemented.')

    def init_training(self, cfg: Dict) -> None:

        self.logger.info('Initializing training.')

        self.num_epochs = cfg['TRAIN.NUM_EPOCHS']
        self.start_epoch = 0
        self.ckpt_save_strategy = cfg.get('TRAIN.CKPT_SAVE_STRATEGY')
        self.best_metrics = {}
        self.clip_grad_param = cfg.get('TRAIN.CLIP_GRAD_PARAM')
        if self.clip_grad_param is not None:
            self.logger.info('Set clip grad, param: {}'.format(self.clip_grad_param))

        self.train_data_loader = self.build_train_data_loader(cfg)
        self.register_epoch_meter('train/time', 'train', '{:.2f} (s)', plt=False)

        self.optim = self.build_optim(cfg['TRAIN.OPTIM'], self.model)
        self.logger.info('Set optim: {}'.format(self.optim))

        self.build_lr_scheduler(cfg)

        if cfg.has('TRAIN.FINETUNE_FROM'):
            self.load_model(cfg['TRAIN.FINETUNE_FROM'], cfg.get('TRAIN.FINETUNE_STRICT_LOAD', True))
            self.logger.info('Start fine tuning')

        self.load_model_resume()

        if is_master():
            self.tensorboard_writer = SummaryWriter(
                os.path.join(self.ckpt_save_dir, 'tensorboard'),
                purge_step=(self.start_epoch + 1) if self.start_epoch != 0 else None
            )

        if cfg.has('VAL'):
            self.init_validation(cfg)

        if hasattr(cfg, 'TEST'):
            self.init_test(cfg)

    @master_only
    def init_validation(self, cfg: Dict):

        self.logger.info('Initializing validation.')
        self.val_interval = cfg.get('VAL.INTERVAL', 1)
        self.val_data_loader = self.build_val_data_loader(cfg)
        self.register_epoch_meter('val/time', 'val', '{:.2f} (s)', plt=False)

    @master_only
    def init_test(self, cfg: Dict) -> None:

        self.test_interval = cfg['TEST'].get('INTERVAL', 1)
        self.test_data_loader = self.build_test_data_loader(cfg)
        self.register_epoch_meter('test/time', 'test', '{:.2f} (s)', plt=False)

    def train(self, cfg: Dict) -> None:

        self.init_training(cfg)

        train_time_predictor = TimePredictor(self.start_epoch, self.num_epochs)
        self.total_iters = self.iter_per_epoch * self.num_epochs

        epoch_index = 0
        for epoch_index in range(self.start_epoch, self.num_epochs):
            epoch = epoch_index + 1

            if self.check_early_stopping():
                break

            self.on_epoch_start(epoch)
            epoch_start_time = time.time()
            self.model.train()

            data_loader = tqdm(self.train_data_loader) if get_local_rank() == 0 \
                                                        else self.train_data_loader

            for iter_index, data in enumerate(data_loader):

                loss = self.train_iters(epoch, iter_index, data)
                if loss is not None:
                    self.backward(loss)

            if self.scheduler is not None:
                self.scheduler.step()

            epoch_end_time = time.time()
            self.update_epoch_meter('train/time', epoch_end_time - epoch_start_time)
            self.on_epoch_end(epoch)

            expected_end_time = train_time_predictor.get_expected_end_time(epoch)

            if epoch < self.num_epochs:
                self.logger.info('The estimated training finish time is {}'.format(
                    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(expected_end_time))))

        self.logger.info('The training finished at {}'.format(
            time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        ))

        self.on_training_end(cfg=cfg, train_epoch=epoch_index + 1)

    @torch.no_grad()
    @master_only
    def validate(self, cfg: Dict = None, train_epoch: Optional[int] = None):

        if train_epoch is None:
            self.init_validation(cfg)

        self.logger.info('Start validation.')

        self.on_validating_start(train_epoch)

        val_start_time = time.time()
        self.model.eval()

        data_iter = tqdm(self.val_data_loader)

        for iter_index, data in enumerate(data_iter):
            self.val_iters(train_epoch, iter_index, data)

        val_end_time = time.time()
        self.update_epoch_meter('val/time', val_end_time - val_start_time)
        self.print_epoch_meters('val')
        if train_epoch is not None:
            self.plt_epoch_meters('val', train_epoch // self.val_interval)

        self.on_validating_end(train_epoch)

    @torch.no_grad()
    @master_only
    def test_pipeline(self, cfg: Optional[Dict] = None, train_epoch: Optional[int] = None, save_metrics: bool = False, save_results: bool = False) -> None:

        if train_epoch is None and cfg is not None:
            self.init_test(cfg)

        self.on_test_start()

        test_start_time = time.time()
        self.model.eval()

        self.test(train_epoch=train_epoch, save_results=save_results, save_metrics=save_metrics)

        test_end_time = time.time()
        self.update_epoch_meter('test/time', test_end_time - test_start_time)

        self.print_epoch_meters('test')
        if train_epoch is not None:
            self.plt_epoch_meters('test', train_epoch // self.test_interval)

        if save_results:
            self.logger.info(f'Test results_1 saved to {os.path.join(self.ckpt_save_dir, "test_results.npz")}.')
        if save_metrics:
            self.logger.info(f'Test metrics saved to {os.path.join(self.ckpt_save_dir, "test_metrics.json")}.')

        self.on_test_end()

    @abstractmethod
    def train_iters(self, epoch: int, iter_index: int, data: Union[torch.Tensor, Tuple]) -> torch.Tensor:

        pass

    def val_iters(self, epoch: int, iter_index: int, data: Union[torch.Tensor, Tuple]) -> None:

        raise NotImplementedError()

    def test(self, train_epoch: Optional[int] = None, save_metrics: bool = False, save_results: bool = False) -> None:

        raise NotImplementedError('test method must be implemented.')

    def on_epoch_start(self, epoch: int):

        self.logger.info('Epoch {:d} / {:d}'.format(epoch, self.num_epochs))
        if self.scheduler is not None:
            self.update_epoch_meter('train/lr', self.scheduler.get_last_lr()[0])

        sampler = self.train_data_loader.sampler
        if torch.distributed.is_initialized() and isinstance(sampler, DistributedSampler) and sampler.shuffle:
            sampler.set_epoch(epoch)

    def on_epoch_end(self, epoch: int) -> None:

        self.print_epoch_meters('train')
        self.plt_epoch_meters('train', epoch)
        if self.val_data_loader is not None and epoch % self.val_interval == 0:
            self.validate(train_epoch=epoch)
        if self.test_data_loader is not None and epoch % self.test_interval == 0:
            self.test_pipeline(train_epoch=epoch)
        self.save_model(epoch)
        self.reset_epoch_meters()

    @master_only
    def on_validating_start(self, train_epoch: Optional[int]):

        pass

    @master_only
    def on_validating_end(self, train_epoch: Optional[int]):

        pass

    @master_only
    def on_test_start(self) -> None:

        pass

    @master_only
    def on_test_end(self) -> None:

        pass

    def on_training_end(self, cfg: Dict, train_epoch: Optional[int] = None):

        if is_master():
            self.tensorboard_writer.close()

        if hasattr(cfg, 'TEST'):
            best_model_path = os.path.join(
                self.ckpt_save_dir,
                '{}_best_val_{}.pt'.format(self.model_name, self.target_metrics.replace('/', '_'))
            )
            self.logger.info('Evaluating the best model on the test set.')
            self.load_model(ckpt_path=best_model_path, strict=True)
            self.test_pipeline(cfg=cfg, train_epoch=train_epoch, save_metrics=True, save_results=True)

    def load_model(self, ckpt_path: str = None, strict: bool = True) -> None:

        try:
            checkpoint_dict = load_ckpt(self.ckpt_save_dir, ckpt_path=ckpt_path, logger=self.logger)
            if isinstance(self.model, DDP):
                self.model.module.load_state_dict(checkpoint_dict['model_state_dict'], strict=strict)
            else:
                self.model.load_state_dict(checkpoint_dict['model_state_dict'], strict=strict)
        except (IndexError, OSError) as e:
            raise OSError('Ckpt file does not exist') from e

    def load_model_resume(self, strict: bool = True):

        try:
            checkpoint_dict = load_ckpt(self.ckpt_save_dir, logger=self.logger)
            if isinstance(self.model, DDP):
                self.model.module.load_state_dict(checkpoint_dict['model_state_dict'], strict=strict)
            else:
                self.model.load_state_dict(checkpoint_dict['model_state_dict'], strict=strict)
            self.optim.load_state_dict(checkpoint_dict['optim_state_dict'])
            self.start_epoch = checkpoint_dict['epoch']
            if checkpoint_dict.get('best_metrics') is not None:
                self.best_metrics = checkpoint_dict['best_metrics']
            if self.scheduler is not None:
                self.scheduler.last_epoch = checkpoint_dict['epoch']
            self.logger.info('Resume training')
        except (IndexError, OSError, KeyError):
            pass

    def backward(self, loss: torch.Tensor):

        self.optim.zero_grad()
        loss.backward()
        if self.clip_grad_param is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), **self.clip_grad_param)
        self.optim.step()

    @master_only
    def save_model(self, epoch: int):

        model = self.model.module if isinstance(self.model, DDP) else self.model
        ckpt_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optim_state_dict': self.optim.state_dict(),
            'best_metrics': self.best_metrics
        }

        last_ckpt_path = self.get_ckpt_path(epoch - 1)
        backup_last_ckpt(last_ckpt_path, epoch, self.ckpt_save_strategy)

        ckpt_path = self.get_ckpt_path(epoch)
        save_ckpt(ckpt_dict, ckpt_path, self.logger)

        if epoch % 10 == 0 or epoch == self.num_epochs:
            clear_ckpt(self.ckpt_save_dir)

    @master_only
    def save_best_model(self, epoch: int, metric_name: str, greater_best: bool = True):

        metric = self.meter_pool.get_avg(metric_name)
        best_metric = self.best_metrics.get(metric_name)
        if best_metric is None or (metric > best_metric if greater_best else metric < best_metric):
            self.best_metrics[metric_name] = metric
            model = self.model.module if isinstance(self.model, DDP) else self.model
            ckpt_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optim_state_dict': self.optim.state_dict(),
                'best_metrics': self.best_metrics
            }
            ckpt_path = os.path.join(
                self.ckpt_save_dir,
                '{}_best_{}.pt'.format(self.model_name, metric_name.replace('/', '_'))
            )
            save_ckpt(ckpt_dict, ckpt_path, self.logger)
            self.current_patience = self.early_stopping_patience
        else:
            if self.early_stopping_patience is not None:
                self.current_patience -= 1

    def init_logger(self, logger: logging.Logger = None, logger_name: str = None,
                    log_file_name: str = None, log_level: int = logging.INFO) -> None:

        if logger is not None:
            self.logger = logger
        elif logger_name is not None:
            if log_file_name is not None:
                log_file_name = '{}_{}.log'.format(log_file_name, time.strftime('%Y%m%d%H%M%S', time.localtime()))
                log_file_path = os.path.join(self.ckpt_save_dir, log_file_name)
            else:
                log_file_path = None
            self.logger = get_logger(logger_name, log_file_path, log_level)
        else:
            raise TypeError('At least one of logger and logger_name is not None')

    def get_ckpt_path(self, epoch: int) -> str:

        epoch_str = str(epoch).zfill(len(str(self.num_epochs)))
        ckpt_name = '{}_{}.pt'.format(self.model_name, epoch_str)
        return os.path.join(self.ckpt_save_dir, ckpt_name)

    def check_early_stopping(self) -> bool:
        if self.early_stopping_patience is not None and self.current_patience <= 0:
            self.logger.info('Early stopping.')
            return True
        return False

    @master_only
    def register_epoch_meter(self, name, meter_type, fmt='{:f}', plt=True) -> None:
        if self.meter_pool is None:
            self.meter_pool = MeterPool()
        self.meter_pool.register(name, meter_type, fmt, plt)

    @master_only
    def update_epoch_meter(self, name, value, n=1) -> None:
        self.meter_pool.update(name, value, n)

    @master_only
    def print_epoch_meters(self, meter_type) -> None:
        self.meter_pool.print_meters(meter_type, self.logger)

    @master_only
    def plt_epoch_meters(self, meter_type, step) -> None:
        self.meter_pool.plt_meters(meter_type, step, self.tensorboard_writer)

    @master_only
    def reset_epoch_meters(self) -> None:
        self.meter_pool.reset()
