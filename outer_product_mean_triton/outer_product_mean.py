from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import Module
import triton
import triton.language as tl

#DEVICE = triton.runtime.driver.active.get_active_torch_device()

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

    def forward(self, ctx, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)

        output = torch.zeros((a.size(-1), b.size(-1)))
        assert a.size(-2) == b.size(-2)
        
        S, M = a.shape
        _, N = b.shape
        #assert a.device == DEVICE and b.device == DEVICE and output.device == DEVICE

        MAX_FUSED_SIZE = 65536 // a.element_size()
        BLOCK_SIZE = MAX_FUSED_SIZE // (M * N) 
        print(f"BLOCK_SIZE: {BLOCK_SIZE}")

        # 1D Grid across sequence
        def grid(meta):
            return triton.cdiv(a.size(-2), meta["BLOCK_SIZE"]),

        _mean_outer_product_fwd[grid](a, b, output, a.size(-2), BLOCK_SIZE)
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
    A,  # pointer to first input A (S, M)
    B,  # pointer to second input B (S, N)
    Output,  # pointer to output vector O (M, N)
    S,  # Averaged dimension
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)

    # Select tile of A and B across S
    print(A)
    offsets = row * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < S
    a = tl.load(A + offsets, mask=mask, other=0).to(tl.float32)  # (BLOCK_SIZE, M)
    b = tl.load(B + offsets, mask=mask, other=0).to(tl.float32)  # (BLOCK_SIZE, N)

    #print(a)

    # Compute outer product
    a = tl.expand_dims(a, axis=1)
    b = tl.expand_dims(b, axis=0)
    outer_product = tl.dot(a, b)  # (BLOCK_SIZE, M, N)
    averaged_outer_product = tl.sum(outer_product, 0) / S  # (M, N)
    # TODO: How to do this sum into shared entries of output tensor.
    tl.store(averaged_outer_product, Output, mask=mask)
