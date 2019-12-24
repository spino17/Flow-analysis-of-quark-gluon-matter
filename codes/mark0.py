import numpy as np
import torch
from torch import nn
import torch.optim as optim


def gaussian_func(x, mean, cov_mat):
    pi = 3.1415926535897932384626433
    num_dim = torch.tensor(2)  # number of dimensions
    print(x - mean)
    print(mean.size())
    val = (
        -1
        / 2
        * torch.mm(
            x - mean, torch.mm(torch.inverse(cov_mat), torch.transpose(x - mean, 0, 1))
        )
    )
    return (
        torch.exp(val).float()
        / torch.sqrt(torch.pow(2 * pi, num_dim) * torch.det(cov_mat)).float()
    )


x = torch.randn(3, 3)
x.requires_grad = True

y = torch.randn(3, 3)
y.requires_grad = True

z = torch.mean(torch.mm(x, y))

optimizer = optim.SGD([x, y], lr=0.01)
optimizer.zero_grad()
z.backward()
optimizer.step()

loc = torch.tensor([[2.0, 3.0], [1000.0, 2000.0]], requires_grad=True)
cov_mat = torch.tensor(
    [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]], requires_grad=True
)
p = torch.distributions.multivariate_normal.MultivariateNormal(loc, cov_mat)
r = p.sample()
print(torch.exp(p.log_prob(r)))


print(gaussian_func(r.view(1, -1), loc.view(1, -1), cov_mat))

n = 11
z = torch.arange(1, 2 * n - 1, 2)


