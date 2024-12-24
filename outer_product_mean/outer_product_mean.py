from torch import Tensor
from torch.nn import Module
from .kernel import FastOuterProductMeanFunction


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

    def forward(
        self,
        A: Tensor,
        B: Tensor,
        use_triton_kernel: bool = False,
        use_batched_matmul: bool = False,
    ) -> Tensor:
        if use_triton_kernel:
            return FastOuterProductMeanFunction.apply(A, B)
        if use_batched_matmul:
            raise Exception("UNIMPLEMENTED")
        return self._naive(A, B)

    def _naive(self, A: Tensor, B: Tensor) -> Tensor:
        return (A.transpose(-2, -1) @ B) / A.size(-2)
