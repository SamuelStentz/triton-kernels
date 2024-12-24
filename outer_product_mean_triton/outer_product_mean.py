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
    Sequence dimension must be a power of 2 and <= 65536 // element_size.

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
        MAX_FUSED_SIZE = 65536 // A.element_size()

        assert Batch == Batch_2
        assert S == S_2
        assert S == triton.next_power_of_2(S)
        assert S <= MAX_FUSED_SIZE

        # 2D Grid
        def grid(_):
            return (M, N)

        Output = torch.zeros((Batch, M, N), device="cuda:0").contiguous()
        # assert A.device == DEVICE and B.device == DEVICE and Output.device == DEVICE
        A = torch.transpose(A, -1, -2)
        B = torch.transpose(B, -1, -2)

        # Complete every batch.
        for b in range(Batch):
            A_slice = A[b]
            B_slice = A[b]
            Output_slice = Output[b].contiguous()
            print(f"Output_slice shape: {Output_slice.shape}")
            _mean_outer_product_fwd[grid](
                A_slice, B_slice, Output_slice, A_slice.stride(0), B_slice.stride(0), Output_slice.stride(0), S
            )
        return Output


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
