from __future__ import annotations

from torch import einsum, Tensor
from torch.nn import Module


class OuterProductMean(Module):
    """
    Compute mean_s(a_si ⊗ b_sj), the mean of outer products over the sequence dimension.

    Args:
        a: Tensor of shape [*, s, i]
        b: Tensor of shape [*, s, j]

    Returns:
        Tensor of shape [*, i, j], the mean of the outer products.
    """

    def __init__(self):
        super().__init__()

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        a = a.transpose(-1, -2)
        b = b.transpose(-1, -2)
        outer_products = einsum("...is,...js->...ijs", a, b)
        return outer_products.mean(dim=-1)
