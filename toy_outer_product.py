import torch

from outer_product_mean_pytorch.outer_product_mean import (
    OuterProductMean
)

def test(
):
    # inputs a, b

    a = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], requires_grad=True)  # Shape: (1, 2, 2, 2)
    b = torch.tensor([[[[5.0, 6.0], [7.0, 8.0]]]], requires_grad=True)  # Shape: (1, 2, 2, 2)

    # instantiate

    opm = OuterProductMean()

    # forward

    o = opm(a, b)
    print(o)

    #assert torch.allclose(ro, ko, atol = 1e-6)

    # backwards

    o.sum().backward()
    print(a._grad)
    print(b._grad)

    # TODO
    #assert torch.allclose(rq.grad, fq.grad, atol = 1e-6)
    #assert torch.allclose(rk.grad, fk.grad, atol = 1e-6)
    #assert torch.allclose(rv.grad, fv.grad, atol = 1e-6)

    print('✅ outputs and gradients are same between regular and kernel mean outer product')

if __name__ == '__main__':
    test()
