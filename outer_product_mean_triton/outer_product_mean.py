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

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        S, M = a.shape
        S_2, N = b.shape
        MAX_FUSED_SIZE = 65536 // a.element_size()

        assert S == S_2
        assert S == triton.next_power_of_2(S)
        assert S <= MAX_FUSED_SIZE

        # 2D Grid
        def grid(_):
            return (M, N)

        a = a.contiguous()
        b = b.contiguous()
        output = torch.zeros((M, N), device="cuda:0").contiguous()
        
        # assert a.device == DEVICE and b.device == DEVICE and output.device == DEVICE
        _mean_outer_product_fwd[grid](
            a, b, output, S, a.stride(0), b.stride(0), output.stride(0)
        )
        return output


@triton.jit
def _mean_outer_product_fwd(
    A,  # pointer to first input A (S, M)
    B,  # pointer to second input B (S, N)
    Output,  # pointer to output vector O (M, N)
    a_stride: tl.constexpr,
    b_stride: tl.constexpr,
    o_stride: tl.constexpr,
    S: tl.constexpr,
):
    # Map the program id to Output_ij to compute
    i = tl.program_id(0)
    j = tl.program_id(1)

    # Select A_i, B_j
    a_offsets = tl.arange(0, S) * a_stride + i
    b_offsets = tl.arange(0, S) * b_stride + j
    a = tl.load(A + a_offsets).to(tl.float32)
    b = tl.load(B + b_offsets).to(tl.float32)

    # Compute Output_ij = mean(a ∘ b)
    result = tl.sum(a * b, axis=0) / S
    o = Output + i * o_stride + j
    tl.store(o, result)


@triton.jit
def _mean_outer_product_bwd_dA(
    DA,  # pointer to the A gradient
    DO,  # pointer to the O gradient
    A,  # pointer to first input A (M, S)
    B,  # pointer to second input B (N, S)
    S: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
):
    print("_mean_outer_product_bwd_dA unimplemented")


@triton.jit
def _mean_outer_product_bwd_dB(
    DB,  # pointer to the A gradient
    DO,  # pointer to the O gradient
    A,  # pointer to first input A (M, S)
    B,  # pointer to second input B (N, S)
    S: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
):
    print("_mean_outer_product_bwd_dA unimplemented")
