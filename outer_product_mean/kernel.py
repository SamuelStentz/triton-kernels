import torch
import triton
import triton.language as tl
from torch import Tensor


class FastOuterProductMeanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A: Tensor, B: Tensor):
        ctx.save_for_backward(A, B)

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

        # assert a.device == DEVICE and b.device == DEVICE and output.device == DEVICE
        Output = torch.zeros((Batch, M, N), device="cuda").contiguous()
        A = torch.transpose(A, -1, -2).contiguous()
        B = torch.transpose(B, -1, -2).contiguous()
        for b in range(0, Batch, 1):
            A_slice = A[b]
            B_slice = B[b]
            Output_slice = Output[b]
            _mean_outer_product_fwd[grid](
                A_slice,
                B_slice,
                Output_slice,
                A_slice.stride(0),
                B_slice.stride(0),
                Output_slice.stride(0),
                S,
            )
        return Output

    @staticmethod
    def backward(ctx, dO):
        A, B = ctx.saved_tensors
        S = A.shape[-2]

        # ∇_A = 1/s (∇_O @ B^T)^T
        dA = (dO @ B.transpose(-1, -2)) / S

        # ∇_B = 1/s (∇_O^T @ A^T)^T
        dB = (dO.transpose(-1, -2) @ A.transpose(-1, -2)) / S

        return dA.transpose(-1, -2), dB.transpose(-1, -2)


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
