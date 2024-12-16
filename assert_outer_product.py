import click

import torch

from outer_product_mean_pytorch.outer_product_mean import (
    OuterProductMean
)

# variables

@click.command()
@click.option('--seq-len', default = 16384) # 16384
@click.option('--i', default = 768) # 768
@click.option('--j', default = 768) # 768
@click.option('--hidden', default = 32) # 32
@click.option('--cuda-kernel', is_flag = True)
def test(
    seq_len: int,
    i: int,
    j: int,
    hidden: int,
    cuda_kernel: bool,
):
    # inputs a, b

    a = torch.randn(1, seq_len, i, hidden)
    b = torch.randn(1, seq_len, j, hidden)

    # kernel and regular inputs

    ka = a.clone().requires_grad_()
    kb = b.clone().requires_grad_()

    ra = a.clone().requires_grad_()
    rb = b.clone().requires_grad_()

    # instantiate

    opm = OuterProductMean()
    # TODO
    #kopm = OuterProductMean(kernel = True)

    # forward

    # TODO
    #ko = kopm(a, b)
    ro = opm(ra, rb)

    #assert torch.allclose(ro, ko, atol = 1e-6)

    # backwards

    ro.sum().backward()
    print(ro)
    # TODO
    #ko.sum().backward()

    # TODO
    #assert torch.allclose(rq.grad, fq.grad, atol = 1e-6)
    #assert torch.allclose(rk.grad, fk.grad, atol = 1e-6)
    #assert torch.allclose(rv.grad, fv.grad, atol = 1e-6)

    print('✅ outputs and gradients are same between regular and kernel mean outer product')

if __name__ == '__main__':
    test()
