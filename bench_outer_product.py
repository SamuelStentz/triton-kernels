import torch
import triton

from outer_product_mean_triton.outer_product_mean import Fast_OuterProductMean
from outer_product_mean_pytorch.outer_product_mean import OuterProductMean


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['seq_len'],
        x_vals=[2**i for i in range(5, 15, 1)],
        x_log=True,
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='outer-product-forward-performance',
        args={},
    ))
def benchmark(seq_len, provider):
    a = torch.rand(seq_len, 1024, dtype=torch.float32).cuda()
    b = torch.rand(seq_len, 1, dtype=torch.float32).cuda()
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        opm = OuterProductMean()
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: opm(a, b), quantiles=quantiles)
    if provider == 'triton':
        opm = Fast_OuterProductMean()
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: opm(a, b), quantiles=quantiles)
    gbps = lambda ms: (a.numel() + a.shape[1]*b.shape[1] + b.numel()) * a.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == '__main__':
    benchmark.run(print_data=True, show_plots=True, save_path="performance")
