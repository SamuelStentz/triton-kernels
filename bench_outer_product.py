import torch
import triton
import gc

from outer_product_mean.outer_product_mean import OuterProductMean


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["seq_len"],
        x_vals=[2**i for i in range(5, 16, 1)],
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "naive", "batched"],
        line_names=["Triton", "Naive", "Batched"],
        styles=[("blue", "-"), ("green", "-"), ("red", "-")],
        ylabel="GB/s",
        plot_name="outer-product-performance",
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
            batch, seq_len, m, dtype=torch.float32, device="cuda", requires_grad=True
        )
        B = torch.rand(
            batch, seq_len, n, dtype=torch.float32, device="cuda", requires_grad=True
        )
        dO = torch.rand(batch, m, n, dtype=torch.float32, device="cuda")
        opm = OuterProductMean()

        quantiles = [0.5, 0.2, 0.8]

        if provider == "naive":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: forward_backwards(opm, A, B, dO),  # noqa: F821
                quantiles=quantiles,
            )
        if provider == "triton":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: forward_backwards(opm, A, B, dO, use_triton_kernel=True),  # noqa: F821
                quantiles=quantiles,
            )
        if provider == "batched":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: forward_backwards(opm, A, B, dO, use_batched_matmul=True),  # noqa: F821
                quantiles=quantiles,
            )
    except Exception as e:
        print(f"Failed: {e}")
        return float("nan"), float("nan"), float("nan")

    ms, max_ms, min_ms = gbps(ms, A, B), gbps(max_ms, A, B), gbps(min_ms, A, B)

    del A
    del B
    del dO
    del opm

    return ms, max_ms, min_ms


def forward_backwards(
    opm,
    A,
    B,
    dO,
    use_triton_kernel: bool = False,
    use_batched_matmul: bool = False,
):
    o = opm(
        A,
        B,
        use_triton_kernel=use_triton_kernel,
        use_batched_matmul=use_batched_matmul,
    )
    o.backward(dO)


def gbps(ms, A, B):
    return (
        2
        * (A.numel() + A.shape[1] * B.shape[1] + B.numel())
        * A.element_size()
        * 1e-09
        / (ms * 0.001)
    )


if __name__ == "__main__":
    benchmark.run(print_data=True, show_plots=True, save_path="performance")
