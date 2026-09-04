"""Muon-style optimizers.

This module provides:

* Muon      - the bare Muon optimizer (Newton-Schulz orthogonalization of the
              momentum buffer) for matrix-valued parameters.
* MuonOpti - a hybrid optimizer that applies Muon to weight matrices
              (ndim == 2, or ndim == 4) and AdamW to the
              remaining parameters (1D biases, group-norm scales).

Muon is intended for parameters whose gradient is a matrix (e.g. conv/linear
weights); the orthogonalized update keeps the Newton-Schulz iterates
approximately orthonormal, which makes the step behave like a scaled gradient
descent on a well-conditioned manifold.
"""

import torch
import torch.optim as optim
from typing import Iterable


def params_group(params: Iterable[torch.nn.Parameter]):
    """Split params into two groups for Muon vs. AdamW.

    group2 (Muon):  2D params, and 4D conv weights
    group1 (AdamW): everything else (1D param vector).
    """
    group1 = []
    group2 = []
    for param in params:
        if not param.requires_grad:
            continue
        if param.ndim == 2 or param.ndim == 4:
            group2.append(param)
        else:
            group1.append(param)
    return group1, group2


@torch.no_grad()
def orthogonalize(matrix: torch.Tensor, steps: int = 5):
    """Newton-Schulz orthogonalization (iterative).

    Arguments
    ---------
    matrix : torch.Tensor
        A 2D gradient matrix (a 4D conv weight is flattened first).
    steps : int
        Number of Newton-Schulz iterations.

    Returns
    -------
    A matrix U with U.T @ U approximately equal to the identity (U is then
    re-scaled to a fixed norm in-place).
    """
    if matrix.dim() == 4:
        out_channels, in_channels, kh, kw = matrix.shape
        matrix = matrix.view(out_channels, -1)  # (out_channels, in_channels * kh * kw)
        is_dim_4 = True
    else:
        is_dim_4 = False
    if matrix.dim() != 2:
        return matrix

    m, n = matrix.shape
    if m < n:
        is_tall = True
    else:
        is_tall = False
        matrix = matrix.T

    a = newtonschulz5(matrix, steps=steps)

    # Normalize the orthonormalized update to a fixed RMS-scaled step size.
    a = a.mul_(0.2 * max(m, n) ** 0.5)

    if not is_tall:
        a = a.T

    if is_dim_4:
        if is_tall:
            a = a.contiguous()
        a = a.view(out_channels, in_channels, kh, kw)
    return a


def newtonschulz5(G: torch.Tensor, steps=5, eps=1e-16):
    """Apply 5 Newton-Schulz iterations (in bf16) to orthonormalize G."""
    X = G.bfloat16()
    a, b, c = (3.4445, -4.7750, 2.0315)  # tuned Newton-Schulz coefficients
    X = X.div_(X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = torch.addmm(A, A, A, beta=b, alpha=c)
        X = torch.addmm(X, B, X, beta=a)
    return X.float()


class Muon(torch.optim.Optimizer):
    """Muon optimizer: orthogonalize the momentum buffer, then update the weights.

    Only parameters with requires_grad and ndim >= 2 are optimized; a plain
    AdamW is better suited to 1D parameter vectors (see MuonOpti).
    """

    def __init__(self,
                 params: Iterable[torch.nn.Parameter],
                 lr: float = 0.02,
                 momentum: float = 0.95,
                 bate: float = 0.999,
                 steps: int = 5,
                 weight_decay: float = 0.0,
                 wd_power: float = 2.0):

        # Keep only trainable, matrix-valued (ndim >= 2) parameters.
        valid_params = []
        for p in params:
            if p.requires_grad and p.dim() >= 2:
                valid_params.append(p)

        if not valid_params:
            raise ValueError("No trainable params with ndim >= 2 found! Muon only optimizes weight matrices.")

        defaults = dict(lr=lr, momentum=momentum, steps=steps, weight_decay=weight_decay, bate=bate, wd_power=wd_power)
        super().__init__(valid_params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta = group['momentum']
            bate = group['bate']
            steps = group['steps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # Initialize the momentum buffer lazily on the first step.
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p)

                momentum_buffer = state['momentum_buffer']

                momentum_buffer.mul_(beta).add_(grad, alpha=1 - beta)

                ortho_grad = orthogonalize(momentum_buffer, steps=steps)

                if group['weight_decay'] > 0:
                    ortho_grad.add_(p.data, alpha=group['weight_decay'])


                p.data.add_(ortho_grad, alpha=-lr)

        return loss

    def zero_grad(self, set_to_none: bool = True):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.zero_()


class MuonOpti(torch.optim.Optimizer):
    """Hybrid optimizer: Muon for weight matrices, AdamW for the rest.

    Args:
        params: iterable of model parameters.
        lr: learning rate (constant during training; Muon gets lr and AdamW
            gets lr / 10).
        weight_decay: L2 weight decay applied by both sub-optimizers.
    """

    def __init__(self, params: Iterable[torch.nn.Parameter], lr=1e-2, weight_decay=0.001):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        params = list(params)
        super(MuonOpti, self).__init__(params, defaults)

        group1, group2 = params_group(params)
        if len(group1) != 0:
            self.adamw = optim.AdamW(group1, lr=lr / 10, weight_decay=weight_decay)
        else:
            self.adamw = None
        self.muon = Muon(group2, lr=lr, weight_decay=weight_decay)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        if self.adamw is not None:
            self.adamw.step()
        self.muon.step()

        return loss

    def zero_grad(self):
        if self.adamw is not None:
            self.adamw.zero_grad()
        self.muon.zero_grad()
