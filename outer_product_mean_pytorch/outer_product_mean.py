from __future__ import annotations

import torch
from torch import einsum, Tensor
from torch.nn import Module


class OuterProductMean(Module):
    """
    Compute mean_s(a_si ⊗ b_sj), the mean of outer products over the sequence dimension.

    Args:
        a: Tensor of shape [batch, s, i]
        b: Tensor of shape [batch, s, j]

    Returns:
        Tensor of shape [batch, i, j], the mean of the outer products.
    """

    def __init__(self):
        super().__init__()

    def forward(self, A: Tensor, B: Tensor) -> Tensor:
        Batch, S, M = A.shape
        Batch_2, S_2, N = B.shape

        assert Batch == Batch_2
        assert S == S_2

        Output = torch.zeros((Batch, M, N), device="cuda:0").contiguous()
        for b in range(Batch):
            Output_slice = einsum("si,sj->sij", A[b], B[b]).mean(dim=-3)
            Output[b] = Output_slice
        return Output
