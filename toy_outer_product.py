import torch

from outer_product_mean.outer_product_mean import OuterProductMean


def test():
    # inputs a, b
    A = torch.tensor(
        [[[1, 2], [3, 4]]],  # shape [1, 3, 2]
        dtype=torch.float32,
        device="cuda",
    )
    B = torch.tensor(
        [[[5, 6], [7, 8]]],  # shape [1, 3, 2]
        dtype=torch.float32,
        device="cuda",
    )
    dO = torch.tensor(
        [[[9, 10], [11, 12]]],  # shape [1, 2, 2]
        dtype=torch.float32,
        device="cuda",
    )

    # kernel and regular inputs
    KA = A.clone().requires_grad_()
    KB = B.clone().requires_grad_()
    RA = A.clone().requires_grad_()
    RB = B.clone().requires_grad_()

    # instantiate
    opm = OuterProductMean()
    kopm = OuterProductMean()

    # forward
    RO = opm.forward(RA, RB)
    KO = kopm.forward(KA, KB, use_triton_kernel=True)
    assert torch.allclose(RO, KO, atol=1e-6)

    # backwards
    RO.backward(dO)
    KO.backward(dO)
    assert torch.allclose(RA.grad, KA.grad, atol=1e-6)
    assert torch.allclose(RB.grad, KB.grad, atol=1e-6)

    print(
        "✅ outputs and gradients are same between regular and kernel mean outer product"
    )


if __name__ == "__main__":
    test()
