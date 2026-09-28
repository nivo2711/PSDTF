import functools
import inspect
import json
import math
import os
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from easydict import EasyDict
from easytorch.utils import master_only
from tqdm import tqdm

from ..metrics import (masked_mae, masked_mape, masked_rmse)
from .base_epoch_runner import BaseEpochRunner

class BaseTimeSeriesForecastingRunner(BaseEpochRunner):

    def __init__(self, cfg: Dict):
        super().__init__(cfg)

        self.need_setup_graph = cfg['MODEL'].get('SETUP_GRAPH', False)

        self.scaler = self.build_scaler(cfg)

        self.loss = cfg['TRAIN']['LOSS']

        self.metrics = cfg.get('METRICS', {}).get('FUNCS', {
                                                            'MAE': masked_mae, 
                                                            'RMSE': masked_rmse, 
                                                            'MAPE': masked_mape,
                                                        })
        self.target_metrics = cfg.get('METRICS', {}).get('TARGET', 'loss')
        self.metrics_best = cfg.get('METRICS', {}).get('BEST', 'min')
        assert self.target_metrics in self.metrics or self.target_metrics == 'loss', f'Target metric {self.target_metrics} not found in metrics.'
        assert self.metrics_best in ['min', 'max'], f'Invalid best metric {self.metrics_best}.'
        self.null_val = cfg.get('METRICS', {}).get('NULL_VAL', np.nan)

        self.cl_param = cfg['TRAIN'].get('CL', None)
        if self.cl_param is not None:
            self.warm_up_epochs = cfg['TRAIN'].CL.get('WARM_EPOCHS', 0)
            self.cl_epochs = cfg['TRAIN'].CL.get('CL_EPOCHS')
            self.prediction_length = cfg['TRAIN'].CL.get('PREDICTION_LENGTH')
            self.cl_step_size = cfg['TRAIN'].CL.get('STEP_SIZE', 1)

        self.if_evaluate_on_gpu = cfg.get('EVAL', EasyDict()).get('USE_GPU', True)
        self.evaluation_horizons = [_ - 1 for _ in cfg.get('EVAL', EasyDict()).get('HORIZONS', [])]
        assert len(self.evaluation_horizons) == 0 or min(self.evaluation_horizons) >= 0, 'The horizon should start counting from 1.'

    def build_scaler(self, cfg: Dict):

        if 'SCALER' in cfg:
            return cfg['SCALER']['TYPE'](**cfg['SCALER']['PARAM'])
        return None

    def setup_graph(self, cfg: Dict, train: bool):

        dataloader = self.build_test_data_loader(cfg=cfg) if not train else self.build_train_data_loader(cfg=cfg)
        data = next(iter(dataloader))
        self.forward(data=data, epoch=1, iter_num=0, train=train)

    def count_parameters(self):

        num_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f'Number of parameters: {num_parameters}')

    def init_training(self, cfg: Dict):

        if self.need_setup_graph:
            self.setup_graph(cfg=cfg, train=True)
            self.need_setup_graph = False

        super().init_training(cfg)
        self.count_parameters()

        self.register_epoch_meter('train/loss', 'train', '{:.4f}')
        for key in self.metrics:
            self.register_epoch_meter(f'train/{key}', 'train', '{:.4f}')

    def init_validation(self, cfg: Dict):

        super().init_validation(cfg)
        self.register_epoch_meter('val/loss', 'val', '{:.4f}')
        for key in self.metrics:
            self.register_epoch_meter(f'val/{key}', 'val', '{:.4f}')

    def init_test(self, cfg: Dict):

        if self.need_setup_graph:
            self.setup_graph(cfg=cfg, train=False)
            self.need_setup_graph = False

        super().init_test(cfg)
        self.register_epoch_meter('test/loss', 'test', '{:.4f}')
        for key in self.metrics:
            self.register_epoch_meter(f'test/{key}', 'test', '{:.4f}')

    def build_train_dataset(self, cfg: Dict):

        if 'DATASET' not in cfg:
            if 'logger' in inspect.signature(cfg['TRAIN']['DATA']['DATASET']['TYPE'].__init__).parameters:
                cfg['TRAIN']['DATA']['DATASET']['PARAM']['logger'] = self.logger
            if 'mode' in inspect.signature(cfg['TRAIN']['DATA']['DATASET']['TYPE'].__init__).parameters:
                cfg['TRAIN']['DATA']['DATASET']['PARAM']['mode'] = 'train'
            dataset = cfg['TRAIN']['DATA']['DATASET']['TYPE'](**cfg['TRAIN']['DATA']['DATASET']['PARAM'])
            self.logger.info(f'Train dataset length: {len(dataset)}')
            batch_size = cfg['TRAIN']['DATA']['BATCH_SIZE']
            iter_per_epoch = math.ceil(len(dataset) / batch_size)
        else:
            dataset = cfg['DATASET']['TYPE'](mode='train', logger=self.logger, **cfg['DATASET']['PARAM'])
            self.logger.info(f'Train dataset length: {len(dataset)}')
            batch_size = cfg['TRAIN']['DATA']['BATCH_SIZE']
            iter_per_epoch = math.ceil(len(dataset) / batch_size)

        return dataset, iter_per_epoch

    def build_val_dataset(self, cfg: Dict):

        if 'DATASET' not in cfg:
            if 'logger' in inspect.signature(cfg['VAL']['DATA']['DATASET']['TYPE'].__init__).parameters:
                cfg['VAL']['DATA']['DATASET']['PARAM']['logger'] = self.logger
            if 'mode' in inspect.signature(cfg['VAL']['DATA']['DATASET']['TYPE'].__init__).parameters:
                cfg['VAL']['DATA']['DATASET']['PARAM']['mode'] = 'valid'
            dataset = cfg['VAL']['DATA']['DATASET']['TYPE'](**cfg['VAL']['DATA']['DATASET']['PARAM'])
            self.logger.info(f'Validation dataset length: {len(dataset)}')
        else:
            dataset = cfg['DATASET']['TYPE'](mode='valid', logger=self.logger, **cfg['DATASET']['PARAM'])
            self.logger.info(f'Validation dataset length: {len(dataset)}')

        return dataset

    def build_test_dataset(self, cfg: Dict):

        if 'DATASET' not in cfg:
            if 'logger' in inspect.signature(cfg['TEST']['DATA']['DATASET']['TYPE'].__init__).parameters:
                cfg['TEST']['DATA']['DATASET']['PARAM']['logger'] = self.logger
            if 'mode' in inspect.signature(cfg['TEST']['DATA']['DATASET']['TYPE'].__init__).parameters:
                cfg['TEST']['DATA']['DATASET']['PARAM']['mode'] = 'test'
            dataset = cfg['TEST']['DATA']['DATASET']['TYPE'](**cfg['TEST']['DATA']['DATASET']['PARAM'])
            self.logger.info(f'Test dataset length: {len(dataset)}')
        else:
            dataset = cfg['DATASET']['TYPE'](mode='test', logger=self.logger, **cfg['DATASET']['PARAM'])
            self.logger.info(f'Test dataset length: {len(dataset)}')

        return dataset

    def curriculum_learning(self, epoch: int = None) -> int:

        if epoch is None:
            return self.prediction_length
        epoch -= 1
        if epoch < self.warm_up_epochs:
            cl_length = self.prediction_length
        else:
            progress = ((epoch - self.warm_up_epochs) // self.cl_epochs + 1) * self.cl_step_size
            cl_length = min(progress, self.prediction_length)
        return cl_length

    def forward(self, data: tuple, epoch: int = None, iter_num: int = None, train: bool = True, **kwargs) -> Dict:

        raise NotImplementedError()

    def metric_forward(self, metric_func, args: Dict) -> torch.Tensor:

        covariate_names = inspect.signature(metric_func).parameters.keys()
        args = {k: v for k, v in args.items() if k in covariate_names}

        if isinstance(metric_func, functools.partial):
            if 'null_val' not in metric_func.keywords and 'null_val' in covariate_names:
                args['null_val'] = self.null_val
            metric_item = metric_func(**args)
        elif callable(metric_func):
            if 'null_val' in covariate_names:
                args['null_val'] = self.null_val
            metric_item = metric_func(**args)
        else:
            raise TypeError(f'Unknown metric type: {type(metric_func)}')
        return metric_item

    def train_iters(self, epoch: int, iter_index: int, data: Union[torch.Tensor, Tuple]):

        iter_num = (epoch - 1) * self.iter_per_epoch + iter_index
        forward_return = self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True)

        if self.cl_param:
            cl_length = self.curriculum_learning(epoch=epoch)
            forward_return['prediction'] = forward_return['prediction'][:, :cl_length, :, :]
            forward_return['target'] = forward_return['target'][:, :cl_length, :, :]
        loss = self.metric_forward(self.loss, forward_return)

        if 'fdc_loss' in forward_return:
            loss = loss + forward_return["fdc_loss"]

        self.update_epoch_meter('train/loss', loss.item())

        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, forward_return)
            self.update_epoch_meter(f'train/{metric_name}', metric_item.item())
        return loss

    def val_iters(self, epoch, iter_index: int, data: Union[torch.Tensor, Tuple]):

        forward_return = self.forward(data=data, epoch=None, iter_num=iter_index, train=False)
        loss = self.metric_forward(self.loss, forward_return)

        if 'fdc_loss' in forward_return:
            loss = loss + forward_return["fdc_loss"]

        self.update_epoch_meter('val/loss', loss.item())
        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, forward_return)
            self.update_epoch_meter(f'val/{metric_name}', metric_item.item())

    def compute_evaluation_metrics(self, returns_all: Dict):

        metrics_results = {}
        for i in self.evaluation_horizons:
            pred = returns_all['prediction'][:, i, :, :]
            real = returns_all['target'][:, i, :, :]

            metrics_results[f'horizon_{i + 1}'] = {}
            metric_repr = ''
            for metric_name, metric_func in self.metrics.items():
                if metric_name.lower() == 'mase':
                    continue
                metric_item = self.metric_forward(metric_func, {'prediction': pred, 'target': real})
                metric_repr += f', Test {metric_name}: {metric_item.item():.4f}'
                metrics_results[f'horizon_{i + 1}'][metric_name] = metric_item.item()
            self.logger.info(f'Evaluate best model on test data for horizon {i + 1}{metric_repr}')

        metrics_results['overall'] = {}
        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, returns_all)
            self.update_epoch_meter(f'test/{metric_name}', metric_item.item())
            metrics_results['overall'][metric_name] = metric_item.item()

        return metrics_results

    @torch.no_grad()
    @master_only
    def test(self, train_epoch: Optional[int] = None, save_metrics: bool = False, save_results: bool = False) -> Dict:

        prediction, target, inputs = [], [], []

        for data in tqdm(self.test_data_loader):
            forward_return = self.forward(data, epoch=None, iter_num=None, train=False)
            loss = self.metric_forward(self.loss, forward_return)

            if 'fdc_loss' in forward_return:
                loss = loss + forward_return["fdc_loss"]

            self.update_epoch_meter('test/loss', loss.item())
            if not self.if_evaluate_on_gpu:
                forward_return['prediction'] = forward_return['prediction'].detach().cpu()
                forward_return['target'] = forward_return['target'].detach().cpu()
                forward_return['inputs'] = forward_return['inputs'].detach().cpu()

            prediction.append(forward_return['prediction'])
            target.append(forward_return['target'])
            inputs.append(forward_return['inputs'])

        prediction = torch.cat(prediction, dim=0)
        target = torch.cat(target, dim=0)
        inputs = torch.cat(inputs, dim=0)

        returns_all = {'prediction': prediction, 'target': target, 'inputs': inputs}
        metrics_results = self.compute_evaluation_metrics(returns_all)

        if save_results:
            test_results = {k: v.cpu().numpy() for k, v in returns_all.items()}
            np.savez(os.path.join(self.ckpt_save_dir, 'test_results.npz'), **test_results)

        if save_metrics:
            with open(os.path.join(self.ckpt_save_dir, 'test_metrics.json'), 'w') as f:
                json.dump(metrics_results, f, indent=4)

        return returns_all

    @master_only
    def on_validating_end(self, train_epoch: Optional[int]):
        greater_best = not self.metrics_best == 'min'
        if train_epoch is not None:
            self.save_best_model(train_epoch, 'val/' + self.target_metrics, greater_best=greater_best)
