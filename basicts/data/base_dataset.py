from dataclasses import dataclass
from typing import List

import numpy as np
from torch.utils.data import Dataset


@dataclass
class BaseDataset(Dataset):

    dataset_name: str
    train_val_test_ratio: List[float]
    mode: str
    input_len: int
    output_len: int
    overlap: bool = False

    def _load_description(self) -> dict:

        raise NotImplementedError("Subclasses must implement this method.")

    def _load_data(self) -> np.ndarray:

        raise NotImplementedError("Subclasses must implement this method.")

    def __len__(self) -> int:

        raise NotImplementedError("Subclasses must implement this method.")

    def __getitem__(self, idx: int) -> dict:

        raise NotImplementedError("Subclasses must implement this method.")
