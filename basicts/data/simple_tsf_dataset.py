import inspect
import json
import logging
from typing import List

import numpy as np
from .base_dataset import BaseDataset


class TimeSeriesForecastingDataset(BaseDataset):

    def __init__(self, dataset_name: str, train_val_test_ratio: List[float], mode: str, input_len: int,
                 output_len: int, overlap: bool = False, logger: logging.Logger = None) -> None:
        assert mode in ['train', 'valid', 'test'], f"Invalid mode: {mode}. Must be one of ['train', 'valid', 'test']."
        super().__init__(dataset_name, train_val_test_ratio, mode, input_len, output_len, overlap)
        self.logger = logger

        self.dataset_name = dataset_name
        self.data_file_path = f'datasets/{dataset_name}/data.dat'
        self.description_file_path = f'datasets/{dataset_name}/desc.json'
        self.description = self._load_description()
        self.data = self._load_data()

    def _load_description(self) -> dict:

        try:
            with open(self.description_file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f'Description file not found: {self.description_file_path}') from e
        except json.JSONDecodeError as e:
            raise ValueError(f'Error decoding JSON file: {self.description_file_path}') from e

    def _load_data(self) -> np.ndarray:

        try:
            data = np.memmap(self.data_file_path, dtype='float32', mode='r', shape=tuple(self.description['shape']))
        except (FileNotFoundError, ValueError) as e:
            raise ValueError(f'Error loading data file: {self.data_file_path}') from e

        total_len = len(data)
        valid_len = int(total_len * self.train_val_test_ratio[1])
        test_len = int(total_len * self.train_val_test_ratio[2])
        train_len = total_len - valid_len - test_len

        minimal_len = self.input_len + self.output_len
        if minimal_len > {'train': train_len, 'valid': valid_len, 'test': test_len}[self.mode]:
            self.overlap = True
            current_frame = inspect.currentframe()
            file_name = inspect.getfile(current_frame)
            line_number = current_frame.f_lineno - 7
            dataset = {'train': 'Training', 'valid': 'Validation', 'test': 'Test'}[self.mode]
            if self.logger is not None:
                self.logger.info(f'{dataset} dataset is too short, enabling overlap. See details in {file_name} at line {line_number}.')
            else:
                print(f'{dataset} dataset is too short, enabling overlap. See details in {file_name} at line {line_number}.')

        if self.mode == 'train':
            offset = self.output_len if self.overlap else 0
            return data[:train_len + offset].copy()
        elif self.mode == 'valid':
            offset_left = self.input_len - 1 if self.overlap else 0
            offset_right = self.output_len if self.overlap else 0
            return data[train_len - offset_left : train_len + valid_len + offset_right].copy()
        else:
            offset = self.input_len - 1 if self.overlap else 0
            return data[train_len + valid_len - offset:].copy()

    def __getitem__(self, index: int) -> dict:
        history_data = self.data[index : index + self.input_len]
        future_data = self.data[index + self.input_len : index + self.input_len + self.output_len]

        return {'inputs': history_data, 'target': future_data}

    def __len__(self) -> int:
        return len(self.data) - self.input_len - self.output_len + 1
