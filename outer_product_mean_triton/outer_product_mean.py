from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import Module
import triton
import triton.language as tl

# DEVICE = triton.runtime.driver.active.get_active_torch_device()


class Fast_OuterProductMean(Module):
    """
    Compute mean_s(a_si ⊗ b_sj), the mean of outer products over the sequence dimension.

    Args:
        a: Tensor of shape [s, i]
        b: Tensor of shape [s, j]

    Returns:
        Tensor of shape [i, j], the mean of the outer products.
    """

    def __init__(self):
        super().__init__()

    def forward(self, ctx, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)

        S, M = a.shape
        S_2, N = b.shape
        MAX_FUSED_SIZE = 65536 // a.element_size()

        assert S == S_2
        assert S == triton.next_power_of_2(S)
        assert S <= MAX_FUSED_SIZE

        # 2D Grid
        def grid(_):
            return (M, N)

        output = torch.zeros((M, N), device="cuda:0").contiguous()
        
        # assert a.device == DEVICE and b.device == DEVICE and output.device == DEVICE
        a = torch.transpose(a, -1, -2).contiguous()
        b = torch.transpose(b, -1, -2).contiguous()
        _mean_outer_product_fwd[grid](
            a, b, output, a.stride(0), b.stride(0), output.stride(0), S
        )
        return output


    def backward(ctx, dO):
        A, B = ctx.saved_tensors
        S = A.shape[-2]

        # ∇_A = 1/s ∇_O * B
        dA = torch.matmul(dO, B) / S

        # ∇_B = 1/s (∇_O)^T * A
        dB = torch.matmul(torch.transpose(dO, -1, -2), A) / S

        return dA, dB


@triton.jit
def _mean_outer_product_fwd(
    A,  # pointer to first input A (M, S)
    B,  # pointer to second input B (N, S)
    Output,  # pointer to output vector O (M, N)
    a_stride,
    b_stride,
    o_stride,
    S: tl.constexpr,
):
    # Map the program id to Output_ij to compute
    i = tl.program_id(0)
    j = tl.program_id(1)

    # Select A_i, B_j
    a_offsets = tl.arange(0, S) + i * a_stride
    b_offsets = tl.arange(0, S) + j * b_stride
    a = tl.load(A + a_offsets).to(tl.float32)
    b = tl.load(B + b_offsets).to(tl.float32)

    # Compute Output_ij = mean(a ∘ b)
    result = tl.sum(a * b, axis=0) / S
    o = Output + i * o_stride + j
    tl.store(o, result)
