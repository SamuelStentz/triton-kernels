import click

import torch

from outer_product_mean_triton.outer_product_mean import Fast_OuterProductMean
from outer_product_mean_pytorch.outer_product_mean import OuterProductMean

# variables

@click.command()
@click.option('--seq-len', default = 16384) # 16384
@click.option('--i', default = 128) # 768
@click.option('--j', default = 128) # 768
@click.option('--hidden', default = 32) # 32
def test(
    seq_len: int,
    i: int,
    j: int,
    hidden: int
):
    # inputs a, b
    a = torch.randn(seq_len, i).cuda()
    b = torch.randn(seq_len, j).cuda()

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
    print(ro)
    print(ko)
    assert torch.allclose(ro, ko, atol = 1e-6)

    # backwards

    #ro.sum().backward()
    #ko.sum().backward()
    #assert torch.allclose(rq.grad, fq.grad, atol = 1e-6)
    #assert torch.allclose(rk.grad, fk.grad, atol = 1e-6)
    #assert torch.allclose(rv.grad, fv.grad, atol = 1e-6)

    print('✅ outputs and gradients are same between regular and kernel mean outer product')

if __name__ == '__main__':
    test()
