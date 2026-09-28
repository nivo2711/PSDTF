import numpy as np
import torch


def masked_mape(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:

    zero_mask = ~torch.isclose(target, torch.tensor(0.0).to(target.device), atol=5e-5)

    if np.isnan(null_val):
        null_mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        null_mask = ~torch.isclose(target, torch.tensor(null_val).to(target.device), atol=eps)

    mask = (zero_mask & null_mask).float()

    mask /= torch.mean(mask)
    mask = torch.nan_to_num(mask)

    loss = torch.abs((prediction - target) / target)
    loss *= mask
    loss = torch.nan_to_num(loss)

    return torch.mean(loss)
