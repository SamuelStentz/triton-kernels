from __future__ import annotations

from torch import einsum, Tensor
from torch.nn import Module

from beartype import beartype


class OuterProductMean(Module):
    """
    Compute mean_s(a_sih ⊗ b_sjh), the mean of outer products over the sequence dimension.

    Args:
        a: Tensor of shape [*, s, i, h]
        b: Tensor of shape [*, s, j, h]

    Returns:
        Tensor of shape [*, i, j, h], the mean of the outer products.
    """
    
    @beartype
    def __init__(
        self
    ):
        super().__init__()

    @beartype
    def forward(
        self,
        a: Tensor,
        b: Tensor
    ) -> Tensor:
        a = a.transpose(-2, -3)
        b = b.transpose(-2, -3)
        outer_products = einsum('...ish,...jsh->...ijsh', a, b)
        return outer_products.mean(dim=-2)
