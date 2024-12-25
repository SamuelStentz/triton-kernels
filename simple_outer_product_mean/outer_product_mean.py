import torch
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

    Key Concepts:
        mean_s(a_si ⊗ b_sj) = 1/s * A_T @ B
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
        Batch, S, _ = A.shape
        Batch_2, S_2, _ = B.shape
        assert Batch == Batch_2
        assert S == S_2

        if use_triton_kernel:
            return FastOuterProductMeanFunction.apply(A, B)
        if use_batched_matmul:
            return self._batched_matmul(A, B)
        return self._naive(A, B)

    def _naive(self, A: Tensor, B: Tensor) -> Tensor:
        return torch.einsum("...sm,...sn->...smn", A, B).mean(axis=-3)

    def _batched_matmul(self, A: Tensor, B: Tensor) -> Tensor:
        return torch.bmm(A.transpose(-2, -1), B) / A.size(-2)

    # Experimentally found to be no better than _batched_matmul.
    def _doubly_batched_matmul(self, A: Tensor, B: Tensor) -> Tensor:
        Batch, S, M = A.shape
        _, _, N = B.shape

        # TODO: expose this somehow, dynamic based on output (?)
        S_STRIDE = min(S, 64)
        assert S % S_STRIDE == 0
        NUM_BATCH = S // S_STRIDE

        A_reshaped = A.reshape(Batch, NUM_BATCH, S_STRIDE, M)
        B_reshaped = B.reshape(Batch, NUM_BATCH, S_STRIDE, N)
        A_batched = A_reshaped.contiguous().reshape(Batch * NUM_BATCH, S_STRIDE, M)
        B_batched = B_reshaped.contiguous().reshape(Batch * NUM_BATCH, S_STRIDE, N)

        O_batched = torch.bmm(A_batched.transpose(-1, -2), B_batched)
        return O_batched.reshape(Batch, NUM_BATCH, M, N).sum(dim=-3) / S
