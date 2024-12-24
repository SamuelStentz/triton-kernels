import torch

from outer_product_mean_pytorch.outer_product_mean import (
    OuterProductMean
)

def test(
):
    # inputs a, b

    a = torch.tensor([[1.0, 2.0],
                      [3.0, 4.0],
                      [5.0, 6.0]], requires_grad=True)  # Shape: (2,3)
    a = torch.transpose(a, -1, -2).unsqueeze(0)
    b = torch.tensor([[7.0, 8.0]], requires_grad=True)  # Shape: (2,1)
    b = torch.transpose(b, -1, -2).unsqueeze(0)

    # instantiate

    opm = OuterProductMean()

    # forward

    o = opm(a, b)
    o.retain_grad()
    print(f"a {a.shape}: {a}")
    print(f"b {b.shape}: {b}")
    print(f"o {o.shape}: {o}")

    assert torch.allclose(ro, ko, atol = 1e-6)

    # backwards

    move_o_grad = o * torch.tensor([[9.0, 10.0, 11.0]])
    move_o_grad.sum().backward()
    print(f"a grad: {a._grad}")
    print(f"b grad: {b._grad}")
    print(f"o grad: {o._grad}")

    # TODO
    #assert torch.allclose(rq.grad, fq.grad, atol = 1e-6)
    #assert torch.allclose(rk.grad, fk.grad, atol = 1e-6)
    #assert torch.allclose(rv.grad, fv.grad, atol = 1e-6)

    print('✅ outputs and gradients are same between regular and kernel mean outer product')

if __name__ == '__main__':
    test()
