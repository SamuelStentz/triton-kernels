import click

import torch

from simple_outer_product_mean.outer_product_mean import OuterProductMean
# variables


@click.command()
@click.option("--batch", default=16)
@click.option("--seq-len", default=256)  # 16384
@click.option("--m", default=32)  # 768
@click.option("--n", default=16)  # 768
def test(
    batch: int,
    seq_len: int,
    m: int,
    n: int,
):
    # inputs a, b
    A = torch.randn(batch, seq_len, m).cuda()
    B = torch.randn(batch, seq_len, n).cuda()
    dO = torch.randn(batch, m, n).cuda()

    # kernel and regular inputs
    BA = A.clone().requires_grad_()
    BB = B.clone().requires_grad_()
    RA = A.clone().requires_grad_()
    RB = B.clone().requires_grad_()

    # instantiate
    opm = OuterProductMean()
    bopm = OuterProductMean()

    # forward
    RO = opm.forward(RA, RB)
    BO = bopm.forward(BA, BB, use_batched_matmul=True)
    assert torch.allclose(RO, BO, atol=1e-6)

    # backwards
    RO.backward(dO)
    BO.backward(dO)
    assert torch.allclose(RA.grad, BA.grad, atol=1e-6)
    assert torch.allclose(RB.grad, BB.grad, atol=1e-6)

    print(
        "✅ outputs and gradients are same between regular and kernel mean outer product"
    )

if __name__ == "__main__":
    test()
