import torch
import triton

from outer_product_mean.outer_product_mean import OuterProductMean


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["seq_len"],
        x_vals=[2**i for i in range(5, 15, 1)],
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "Torch"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="GB/s",
        plot_name="outer-product-performance",
        args={},
    )
)
def benchmark(seq_len, provider):
    batch, m, n = 1, 128, 128
    A = torch.rand(
        batch, seq_len, m, dtype=torch.float32, device="cuda", requires_grad=True
    )
    B = torch.rand(
        batch, seq_len, n, dtype=torch.float32, device="cuda", requires_grad=True
    )
    dO = torch.rand(batch, m, n, dtype=torch.float32, device="cuda")
    opm = OuterProductMean()

    quantiles = [0.5, 0.2, 0.8]

    def forward_backwards(
        opm,
        a,
        b,
        do,
        use_triton_kernel: bool = False,
        use_batched_matmul: bool = False,
    ):
        o = opm(
            a,
            b,
            use_triton_kernel=use_triton_kernel,
            use_batched_matmul=use_batched_matmul,
        )
        o.backward(do)

    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: forward_backwards(opm, A, B, dO), quantiles=quantiles
        )
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: forward_backwards(opm, A, B, dO, use_triton_kernel=True),
            quantiles=quantiles,
        )

    def gbps(ms):
        return (
            2
            * (A.numel() + A.shape[1] * B.shape[1] + B.numel())
            * A.element_size()
            * 1e-09
            / (ms * 0.001)
        )

    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    benchmark.run(print_data=True, show_plots=True, save_path="performance")
