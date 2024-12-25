import torch
from torch import Tensor
import triton
import gc

from simple_outer_product_mean.outer_product_mean import OuterProductMean


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["seq_len"],
        x_vals=[2**i for i in range(4, 16, 1)],
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "naive", "batched"],
        line_names=["Triton", "Baseline", "Batched"],
        styles=[("blue", "-"), ("green", "-"), ("red", "-")],
        ylabel="Peak Memory Usage (GB)",
        plot_name="Forward Pass Usage",
        args={},
    )
)
def benchmark(seq_len, provider):
    print(f"seq_len: {seq_len}, provider: {provider}")
    try:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        batch, m, n = 32, 768, 768
        A = torch.rand(
            batch, seq_len, m, dtype=torch.float32, device="cuda", requires_grad=False
        )
        B = torch.rand(
            batch, seq_len, n, dtype=torch.float32, device="cuda", requires_grad=False
        )
        opm = OuterProductMean()

        if provider == "naive":
            opm.forward(A, B)
        if provider == "triton":
            opm.forward(A, B, use_triton_kernel=True)
        if provider == "batched":
            opm.forward(A, B, use_batched_matmul=True)
    except Exception as e:
        print(e)
        return float("nan"), float("nan"), float("nan")

    del A
    del B
    del opm

    peak_memory = torch.cuda.max_memory_allocated() / 2**30
    return peak_memory


def forward(
    opm,
    A,
    B,
    use_triton_kernel: bool = False,
    use_batched_matmul: bool = False,
) -> Tensor:
    return opm(
        A,
        B,
        use_triton_kernel=use_triton_kernel,
        use_batched_matmul=use_batched_matmul,
    )


if __name__ == "__main__":
    benchmark.run(print_data=True, show_plots=True, save_path="performance")
