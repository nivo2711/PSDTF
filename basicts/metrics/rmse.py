import numpy as np
import torch

from .mse import masked_mse


def masked_rmse(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:

    return torch.sqrt(masked_mse(prediction=prediction, target=target, null_val=null_val))
