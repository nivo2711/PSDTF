import numpy as np
import torch

def masked_mae(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:

    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        mask = ~torch.isclose(target, torch.tensor(null_val).expand_as(target).to(target.device), atol=eps, rtol=0.0)

    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.nan_to_num(mask)

    loss = torch.abs(prediction - target)
    loss = loss * mask
    loss = torch.nan_to_num(loss)

    return torch.mean(loss)
