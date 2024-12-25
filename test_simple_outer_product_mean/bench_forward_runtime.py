import torch
from torch import Tensor
import triton
import gc

from simple_outer_product_mean.outer_product_mean import OuterProductMean


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["seq_len"],
        x_vals=[2**i for i in range(10, 16, 1)],
        x_log=True,
        y_log=True,
        line_arg="provider",
        line_vals=["triton", "naive", "batched"],
        line_names=["Triton", "Baseline", "Batched"],
        styles=[("blue", "-"), ("green", "-"), ("red", "-")],
        ylabel="Runtime (ms)",
        plot_name="Forward Pass Performance",
        args={},
    )
)
def benchmark(seq_len, provider):
    print(f"seq_len: {seq_len}, provider: {provider}")
    try:
        gc.collect()
        torch.cuda.empty_cache()

        batch, m, n = 32, 768, 768
        A = torch.rand(
            batch, seq_len, m, dtype=torch.float32, device="cuda", requires_grad=False
        )
        B = torch.rand(
            batch, seq_len, n, dtype=torch.float32, device="cuda", requires_grad=False
        )
        opm = OuterProductMean()

        quantiles = [0.5, 0.2, 0.8]

        if provider == "naive":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: forward(opm, A, B),  # noqa: F821
                quantiles=quantiles,
            )
        if provider == "triton":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: forward(opm, A, B, use_triton_kernel=True),  # noqa: F821
                quantiles=quantiles,
            )
        if provider == "batched":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: forward(opm, A, B, use_batched_matmul=True),  # noqa: F821
                quantiles=quantiles,
            )
    except Exception as e:
        print(e)
        return float("nan"), float("nan"), float("nan")

    del A
    del B
    del opm

    return ms, max_ms, min_ms


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
