import torch
from torch import randn, cosh, tanh
from torch.autograd import grad
torch.set_default_tensor_type(torch.DoubleTensor)


"""
Test script demonstating that the gradients calculated in 
Initialized Equilibrium Propagation for Backprop-Free Training
are correct in the limit of small delta_w
"""


def xavier_init(n_in, n_out):
    return (2./(n_in+n_out))**.5 * randn(n_in, n_out, requires_grad=True)


def cosine_similarity(a, b):
    aflat, bflat = a.flatten(), b.flatten()
    return (aflat @ bflat)/torch.sqrt((aflat@aflat) * (bflat @ bflat))


def deriv_tanh(x):
    return 1/cosh(x)**2


n_samples = 10
n_in, n_hid, n_out = (20, 15, 10)
delta_w_scale = 0.0001

# Setup
x = randn(n_samples, n_in)
w1 = xavier_init(n_in, n_hid)
w2 = xavier_init(n_hid, n_out)

delta_w1 = delta_w_scale * randn(n_in, n_hid)
delta_w2 = delta_w_scale * randn(n_hid, n_out)

w1_star = (w1 - delta_w1).detach()
w2_star = (w2 - delta_w2).detach()

s1 = tanh(x @ w1)
s2 = tanh(s1 @ w2)

s1_star = tanh(x @ w1_star)
s2_star = tanh(s1_star @ w2_star)

l1 = .5*((s1 - s1_star) ** 2).sum()
l2 = .5*((s2 - s2_star) ** 2).sum()

# Now, calculate the gradient:
dl1_dw1, = grad(l1, w1, retain_graph=True)
dl2_dw1, = grad(l2, w1)

dl1_dw1_approx = x.t() @ (x @ delta_w1 * deriv_tanh(x @ w1) ** 2)
dl2_dw1_approx = \
    x.t() @ (x @ delta_w1 * deriv_tanh(x @ w1) @ w2 * deriv_tanh(s1 @ w2) ** 2 @ w2.t() * deriv_tanh(x @ w1)) + \
    x.t() @ (s1 @ delta_w2 * deriv_tanh(s1 @ w2) ** 2 @ w2.t() * deriv_tanh(x @ w1))

print(f'S(dl1_dw1, dl1_dw1_approx)={cosine_similarity(dl1_dw1, dl1_dw1_approx)}')
print(f'S(dl2_dw1, dl2_dw1_approx)={cosine_similarity(dl2_dw1, dl2_dw1_approx)}')


assert torch.allclose(dl1_dw1, dl1_dw1_approx, atol=1e-5)
assert torch.allclose(dl2_dw1, dl2_dw1_approx, atol=1e-5)
