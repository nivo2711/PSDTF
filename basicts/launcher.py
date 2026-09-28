import os
import traceback
from typing import Dict, Optional, Union

import easytorch
from easytorch.config import init_cfg
from easytorch.device import set_device_type
from easytorch.utils import get_logger, set_visible_devices


def evaluation_func(cfg: Dict,
                    ckpt_path: str = None,
                    batch_size: Optional[int] = None,
                    strict: bool = True) -> None:

    logger = get_logger('easytorch-launcher')
    logger.info(f"Initializing runner '{cfg['RUNNER']}'")
    runner = cfg['RUNNER'](cfg)

    runner.init_logger(logger_name='easytorch-evaluation', log_file_name='evaluation_log')

    try:
        if batch_size is not None:
            cfg.TEST.DATA.BATCH_SIZE = batch_size
        else:
            assert 'BATCH_SIZE' in cfg.TEST.DATA, 'Batch size must be specified either in the config or as an argument.'

        if ckpt_path is None or not os.path.exists(ckpt_path):
            ckpt_path_auto = os.path.join(runner.ckpt_save_dir, '{}_best_val_{}.pt'.format(runner.model_name, runner.target_metrics.replace('/', '_')))
            logger.info(f'Checkpoint file not found at {ckpt_path}. Loading the best model checkpoint `{ckpt_path_auto}` automatically.')
            if not os.path.exists(ckpt_path_auto):
                raise FileNotFoundError(f'Checkpoint file not found at {ckpt_path}')
            runner.load_model(ckpt_path=ckpt_path_auto, strict=strict)
        else:
            logger.info(f'Loading model checkpoint from {ckpt_path}')
            runner.load_model(ckpt_path=ckpt_path, strict=strict)

        runner.test_pipeline(cfg=cfg, save_metrics=True, save_results=True)

    except BaseException as e:
        runner.logger.error(traceback.format_exc())
        raise e

def launch_evaluation(cfg: Union[Dict, str],
                      ckpt_path: str,
                      device_type: str = 'gpu',
                      gpus: Optional[str] = None,
                      batch_size: Optional[int] = None) -> None:

    logger = get_logger('easytorch-launcher')
    logger.info('Launching EasyTorch evaluation.')

    while isinstance(cfg, str) and cfg.startswith(('./','.\\')):
        cfg = cfg[2:]
    while ckpt_path.startswith(('./','.\\')):
        ckpt_path = ckpt_path[2:]

    cfg = init_cfg(cfg, save=True)

    set_device_type(device_type)

    if device_type != 'cpu':
        set_visible_devices(gpus)

    evaluation_func(cfg, ckpt_path, batch_size)

def launch_training(cfg: Union[Dict, str],
                    gpus: Optional[str] = None,
                    node_rank: int = 0) -> None:

    while isinstance(cfg, str) and cfg.startswith(('./','.\\')):
        cfg = cfg[2:]

    easytorch.launch_training(cfg=cfg, devices=gpus, node_rank=node_rank)
