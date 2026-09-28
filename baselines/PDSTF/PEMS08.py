import os
import sys
import math
import torch
from easydict import EasyDict
sys.path.append(os.path.abspath(__file__ + '/../../..'))

from basicts.metrics import masked_mae, masked_mape, masked_rmse
from basicts.data import TimeSeriesForecastingDataset
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings, load_dataset_desc, load_adj

from .arch import PDSTF

DATA_NAME = 'PEMS08'
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = regular_settings['INPUT_LEN']
OUTPUT_LEN = regular_settings['OUTPUT_LEN']
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']
NORM_EACH_CHANNEL = regular_settings['NORM_EACH_CHANNEL']
RESCALE = regular_settings['RESCALE']
NULL_VAL = regular_settings['NULL_VAL']
MODEL_ARCH = PDSTF
adj_mx, _ = load_adj("datasets/" + DATA_NAME + "/adj_mx.pkl", "normlap")

MODEL_PARAM = {
    "seq_len": INPUT_LEN,
    "pred_len": OUTPUT_LEN,
    "num_nodes": 170,
    "daily_init_path": 'datasets/PEMS08/daily_init.npy',
    "weekly_init_path": 'datasets/PEMS08/weekly_init.npy',
    "adj": torch.tensor(adj_mx[0]),
    "fdc_weight": 0.5,
    "low_frequency_bins": 2,
    "embed_size": 64,
    "hidden_size": 128,
    "fc_hidden_size": 512,
    "time_of_day_size": 288,
    "day_of_week_size": 7
}
NUM_EPOCHS = 100

CFG = EasyDict()
CFG.DESCRIPTION = 'PDSTF traffic flow forecasting on PEMS08'
CFG.GPU_NUM = 1
CFG.RUNNER = SimpleTimeSeriesForecastingRunner

CFG.ENV = EasyDict()
CFG.ENV.SEED = 1

CFG.DATASET = EasyDict()
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = TimeSeriesForecastingDataset
CFG.DATASET.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_val_test_ratio': TRAIN_VAL_TEST_RATIO,
    'input_len': INPUT_LEN,
    'output_len': OUTPUT_LEN,
})

CFG.SCALER = EasyDict()
CFG.SCALER.TYPE = ZScoreScaler
CFG.SCALER.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_ratio': TRAIN_VAL_TEST_RATIO[0],
    'norm_each_channel': NORM_EACH_CHANNEL,
    'rescale': RESCALE,
})

CFG.MODEL = EasyDict()
CFG.MODEL.NAME = MODEL_ARCH.__name__
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2]
CFG.MODEL.TARGET_FEATURES = [0]

CFG.METRICS = EasyDict()
CFG.METRICS.FUNCS = EasyDict({
                                'MAE': masked_mae,
                                'RMSE': masked_rmse,
                                'MAPE': masked_mape
                            })
CFG.METRICS.TARGET = 'MAE'
CFG.METRICS.NULL_VAL = NULL_VAL

CFG.TRAIN = EasyDict()
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints',
    MODEL_ARCH.__name__,
    '_'.join([DATA_NAME, str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)])
)
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.002
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
desc = load_dataset_desc(DATA_NAME)
train_steps = math.ceil(desc["num_time_steps"] * TRAIN_VAL_TEST_RATIO[0])
CFG.TRAIN.LR_SCHEDULER.TYPE = "OneCycleLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
   "pct_start": 0.3,
   "epochs": NUM_EPOCHS,
   "steps_per_epoch": train_steps,
   "max_lr": CFG.TRAIN.OPTIM.PARAM["lr"]
}
CFG.TRAIN.CLIP_GRAD_PARAM = {
    'max_norm': 5.0
}
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 64
CFG.TRAIN.DATA.SHUFFLE = True

CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = 64

CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = 64


CFG.EVAL = EasyDict()

CFG.EVAL.HORIZONS = [3, 6, 12]
CFG.EVAL.USE_GPU = True
