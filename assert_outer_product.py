import click

import torch

from outer_product_mean_triton.outer_product_mean import Fast_OuterProductMean
from outer_product_mean_pytorch.outer_product_mean import OuterProductMean

# variables

@click.command()
@click.option('--seq-len', default = 16384) # 16384
@click.option('--m', default = 32) # 768
@click.option('--n', default = 16) # 768
def test(
    seq_len: int,
    m: int,
    n: int,
):
    # inputs a, b
    a = torch.randn(seq_len, m).cuda()
    b = torch.randn(seq_len, n).cuda()

    # kernel and regular inputs
    ka = a.clone().requires_grad_()
    kb = b.clone().requires_grad_()
    ra = a.clone().requires_grad_()
    rb = b.clone().requires_grad_()

    # instantiate
    opm = OuterProductMean()
    kopm = Fast_OuterProductMean()

    # forward
    ro = opm(ra, rb)
    ko = kopm(ka, kb)
    assert torch.allclose(ro, ko, atol = 1e-6)

    # backwards
    ro.sum().backward()
    ko.sum().backward()
    assert torch.allclose(ra.grad, ka.grad, atol = 1e-6)
    assert torch.allclose(rb.grad, kb.grad, atol = 1e-6)

    print('✅ outputs and gradients are same between regular and kernel mean outer product')

if __name__ == '__main__':
    test()
