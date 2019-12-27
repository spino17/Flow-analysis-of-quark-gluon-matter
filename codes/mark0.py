import numpy as np
import torch
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt


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

loc = torch.tensor([[2.0, 3.0, 5.0], [1000.0, 2000.0, 7000.0]], requires_grad=True)
cov_mat = torch.tensor(
    [
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.1]],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.1]],
    ],
    requires_grad=True,
)
p = torch.distributions.multivariate_normal.MultivariateNormal(loc, cov_mat)
r = p.sample()
print(torch.exp(p.log_prob(r)))

data = np.loadtxt("CorrelationData.txt", skiprows=1)


print(gaussian_func(r.view(1, -1), loc.view(1, -1), cov_mat))

n = 11
z = torch.arange(1, 2 * n - 1, 2)


x = np.arange(0.1, 6.1, 0.2)
y = np.arange(10, 5010, 20)

xx, yy = np.meshgrid(x, y)

plt.plot(xx, yy)


x = torch.randn(3, 3)
z = x * x
optimizer = optim.Adam([x[0], x[1]])
z.backward()
optim.step()


import torch

n = 3
a = torch.arange(n).unsqueeze(1)  # [n, 1]
b = torch.ones(4).view(2, 2)  # [2, 2]

a[:, :, None] * b  # [n, 2, 2]
