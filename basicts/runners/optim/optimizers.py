import inspect
from typing import Optional, Tuple, Union

import torch
from easytorch.device import _DEVICE_TYPE
from easytorch.utils import get_local_rank, get_world_size
from torch import Tensor
from torch.optim import AdamW

__all__ = ['AdamWnanoGPT', 'Muon']

class AdamWnanoGPT(AdamW):
    def __init__(self,
        params,
        lr: Union[float, Tensor] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        *,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        capturable: bool = False,
        differentiable: bool = False,
    ):
        params = [p for p in params if p.requires_grad]
        decay_params = [p for p in params if p.dim() >= 2]
        nodecay_params = [p for p in params if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and _DEVICE_TYPE in ['gpu', 'mlu']
        super().__init__(optim_groups, lr=lr, betas=betas, \
                                eps=eps, amsgrad=amsgrad, maximize=maximize, foreach=foreach, \
                                capturable=capturable, differentiable=differentiable, fused=use_fused)


class Muon(torch.optim.Optimizer):

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, backend='newtonschulz5', backend_steps=5):
        defaults = {'lr': lr, 'momentum': momentum, 'nesterov': nesterov, 'backend': backend, 'backend_steps': backend_steps}
        super().__init__(params, defaults)
        self.zeropower_backends = {'svd': self._zeropower_via_svd, 'newtonschulz5': self._zeropower_via_newtonschulz5}

    def _zeropower_via_svd(self, G):
        U, _, V = G.svd()
        return U @ V.T

    def _zeropower_via_newtonschulz5(self, G, steps=10, eps=1e-7):
        assert len(G.shape) == 2
        a, b, c = (3.4445, -4.7750,  2.0315)
        X = G.bfloat16()
        X /= (X.norm() + eps)
        if G.size(0) > G.size(1):
            X = X.T
        for _ in range(steps):
            A = X @ X.T
            B = A @ X
            X = a * X + b * B + c * A @ B
        if G.size(0) > G.size(1):
            X = X.T
        return X

    def step(self):

        for group in self.param_groups:

            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = self.zeropower_backends[group['backend']]

            total_params = sum(p.numel() for p in group['params'])
            updates_flat = torch.zeros(total_params, device='cuda', dtype=torch.bfloat16)
            curr_idx = 0
            for i, p in enumerate(group['params']):
                if i % get_world_size() == get_local_rank():
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    if group['nesterov']:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_backend(g, steps=group['backend_steps'])
                    g *= max(1, g.size(0)/g.size(1))**0.5
                    updates_flat[curr_idx:curr_idx+p.numel()] = g.flatten()
                curr_idx += p.numel()

            torch.distributed.all_reduce(updates_flat, op=torch.distributed.ReduceOp.SUM)

            curr_idx = 0
            for p in group['params']:
                g = updates_flat[curr_idx:curr_idx+p.numel()].view_as(p.data).type_as(p.data)
                p.data.add_(g, alpha=-lr)
                curr_idx += p.numel()
