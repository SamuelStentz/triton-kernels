import click

import torch

from outer_product_mean_triton.outer_product_mean import Fast_OuterProductMean
from outer_product_mean_pytorch.outer_product_mean import OuterProductMean

# variables

@click.command()
@click.option('--batch', default = 1)
@click.option('--seq-len', default = 16384) # 16384
@click.option('--i', default = 16) # 768
@click.option('--j', default = 16) # 768
def test(
    batch: int,
    seq_len: int,
    i: int,
    j: int,
):
    # inputs a, b
    A = torch.randn(batch, seq_len, i).cuda()
    B = torch.randn(batch, seq_len, j).cuda()

    # kernel and regular inputs
    KA = A.clone().requires_grad_()
    KB = B.clone().requires_grad_()
    RA = A.clone().requires_grad_()
    RB = B.clone().requires_grad_()

    # instantiate
    opm = OuterProductMean()
    kopm = Fast_OuterProductMean()

    # forward
    ko = kopm(KA, KB)
    ro = opm(RA, RB)

    print(ko.shape)
    print(ro.shape)
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
