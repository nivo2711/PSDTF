import time
from functools import partial

import torch


class partial_func(partial):

    def __str__(self):
        return f"partial({self.func.__name__}, {self.keywords})"

def clock(func):

    def clocked(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        print(f"{func.__name__}: {elapsed:.8f}s")
        return result
    return clocked

def check_nan_inf(tensor: torch.Tensor, raise_ex: bool = True) -> tuple:

    nan = torch.any(torch.isnan(tensor))
    inf = torch.any(torch.isinf(tensor))

    if raise_ex and (nan or inf):
        raise ValueError({"nan": nan, "inf": inf})

    return {"nan": nan, "inf": inf}, nan or inf

def remove_nan_inf(tensor: torch.Tensor) -> torch.Tensor:

    tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
    tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)
    return tensor
