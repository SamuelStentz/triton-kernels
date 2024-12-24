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
        plot_name='outer-product-performance',
        args={},
    ))
def benchmark(seq_len, provider):
    m, n = 128, 128
    a = torch.rand(seq_len, m, dtype=torch.float32, device='cuda', requires_grad=True)
    b = torch.rand(seq_len, n, dtype=torch.float32, device='cuda', requires_grad=True)
    do = torch.rand(m, n, dtype=torch.float32, device='cuda')

    quantiles = [0.5, 0.2, 0.8]
    def forward_backwards(opm, a, b, do):
        o = opm(a, b)
        o.backward(do)

    if provider == 'torch':
        opm = OuterProductMean()
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: forward_backwards(opm, a, b, do), quantiles=quantiles)
    if provider == 'triton':
        opm = Fast_OuterProductMean()
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: forward_backwards(opm, a, b, do), quantiles=quantiles)
    def gbps(ms):
        return 2 * (a.numel() + a.shape[1] * b.shape[1] + b.numel()) * a.element_size() * 1e-09 / (ms * 0.001)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == '__main__':
    benchmark.run(print_data=True, show_plots=True, save_path="performance")
