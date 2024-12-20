import click

import torch
import triton

from outer_product_mean_triton.outer_product_mean import Fast_OuterProductMean
from outer_product_mean_pytorch.outer_product_mean import OuterProductMean


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['seq_len'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(5, 14, 1)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='outer-product-mean-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(size, provider):
    # TODO: consider if 768 is really needed.
    a = torch.rand(size, 768, dtype=torch.float32).cuda()
    b = torch.rand(size, 768, dtype=torch.float32).cuda()
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        opm = OuterProductMean()
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: opm(a, b), quantiles=quantiles)
    if provider == 'triton':
        opm = Fast_OuterProductMean()
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: opm(a, b), quantiles=quantiles)
    gbps = lambda ms: 3 * a.numel() * a.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == '__main__':
    benchmark.run(print_data=True, show_plots=True)
