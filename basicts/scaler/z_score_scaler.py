import json

import numpy as np
import torch

from .base_scaler import BaseScaler


class ZScoreScaler(BaseScaler):

    def __init__(self, dataset_name: str, train_ratio: float, norm_each_channel: bool, rescale: bool):

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
            self.mean = np.mean(train_data, axis=0, keepdims=True)
            self.std = np.std(train_data, axis=0, keepdims=True)
            self.std[self.std == 0] = 1.0
        else:
            self.mean = np.mean(train_data)
            self.std = np.std(train_data)
            if self.std == 0:
                self.std = 1.0
        self.mean, self.std = torch.tensor(self.mean), torch.tensor(self.std)

    def transform(self, input_data: torch.Tensor) -> torch.Tensor:

        mean = self.mean.to(input_data.device)
        std = self.std.to(input_data.device)
        input_data[..., self.target_channel] = (input_data[..., self.target_channel] - mean) / std
        return input_data

    def inverse_transform(self, input_data: torch.Tensor) -> torch.Tensor:

        mean = self.mean.to(input_data.device)
        std = self.std.to(input_data.device)
        input_data = input_data.clone()
        input_data[..., self.target_channel] = input_data[..., self.target_channel] * std + mean
        return input_data
