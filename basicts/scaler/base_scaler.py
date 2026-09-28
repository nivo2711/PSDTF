from dataclasses import dataclass

import torch


@dataclass
class BaseScaler:

    dataset_name: str
    train_ratio: float
    norm_each_channel: bool
    rescale: bool

    def transform(self, input_data: torch.Tensor) -> torch.Tensor:

        raise NotImplementedError("Subclasses should implement this method.")

    def inverse_transform(self, input_data: torch.Tensor) -> torch.Tensor:

        raise NotImplementedError("Subclasses should implement this method.")
