import json

import numpy as np
import torch

from .base_scaler import BaseScaler


class MinMaxScaler(BaseScaler):

    def __init__(self, dataset_name: str, train_ratio: float, norm_each_channel: bool = True, rescale: bool = True):

        super().__init__(dataset_name, train_ratio, norm_each_channel, rescale)
        self.target_channel = 0

        description_file_path = f'datasets/{dataset_name}/desc.json'
        with open(description_file_path, 'r') as f:
            description = json.load(f)
        data_file_path = f'datasets/{dataset_name}/data.dat'
        data = np.memmap(data_file_path, dtype='float32', mode='r', shape=tuple(description['shape']))

        train_size = int(len(data) * train_ratio)
        train_data = data[:train_size, :, self.target_channel].copy()

        if norm_each_channel:
            self.min = np.min(train_data, axis=0, keepdims=True)
            self.max = np.max(train_data, axis=0, keepdims=True)
        else:
            self.min = np.min(train_data)
            self.max = np.max(train_data)
        self.min, self.max = torch.tensor(self.min), torch.tensor(self.max)

    def transform(self, input_data: torch.Tensor) -> torch.Tensor:

        _min = self.min.to(input_data.device)
        _max = self.max.to(input_data.device)
        input_data[..., self.target_channel] = (input_data[..., self.target_channel] - _min) / (_max - _min)
        return input_data

    def inverse_transform(self, input_data: torch.Tensor) -> torch.Tensor:

        _min = self.min.to(input_data.device)
        _max = self.max.to(input_data.device)
        input_data[..., self.target_channel] = input_data[..., self.target_channel] * (_max - _min) + _min
        return input_data
