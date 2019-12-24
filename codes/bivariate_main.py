import torch
from torch import nn
import os
import pandas
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import torch.optim as optim


def gaussian_func(x, c_b, mean, cov_mat):
    pi = 3.1415926535897932384626433
    num_dim = mean.size(1)  # number of dimensions
    val = (
        -1
        / 2
        * torch.mm(
            x - mean(c_b),
            torch.mm(torch.inverse(cov_mat(c_b)), torch.transpose(x - mean(c_b), 0, 1)),
        )
    )
    return torch.exp(val) / torch.sqrt(torch.pow(2 * pi, num_dim) * torch.det(cov_mat))

"""
def mean(x):
    mean_coeff = torch.randn(1, num_dim)
    mean_coeff.requires_grad = True
    a_1 = torch.randn(1)
    a_2 = torch.randn(1)
    a_3 = torch.randn(1)

    # declaring them parameters
    a_1.requires_grad = True
    a_2.requires_grad = True
    a_3.requires_grad = True

    return mean_coeff * torch.exp(-1 * a_1 * x - a_2 * (x ** 2) - a_3 * (x ** 3))


def cov_mat(x):
    cov_coeff = torch.randn(num_dim, num_dim)
    cov_coeff.requires_grad = True
    b_1 = torch.randn(1)
    b_2 = torch.randn(1)
    b_3 = torch.randn(1)

    # declaring them parameters
    b_1.requires_grad = True
    b_2.requires_grad = True
    b_3.requires_grad = True

    return cov_coeff * torch.exp(-1 * b_1 * x - b_2 * (x ** 2) - b_3 * (x ** 3))
"""

def integrate(x, num_steps, mean, cov_mat):
    """
    Current implementation of the function uses rectangle rule.
    We can extend this further to use trapezoid rule instead which will be
    more accurate.

    """

    step_value = 1 / num_steps  # dx
    start = 0
    int_sum = 0  # riemannian sum
    for i in range(num_steps):
        end = start + step_value
        mid = (start + end) / 2
        int_sum += gaussian_func(x, mid, mean, cov_mat) * step_value
        start = end

    return int_sum


def mean_squared_error(y_pred, y_true):
    # TODO
    return 0


def cross_entropy_loss(y_pred, y_true):
    # TODO
    return 0


def data_preprocessing(X, y, batch_size, validation_split=0.2):
    dataset = TensorDataset(X, y)
    batch_size = batch_size
    training_length = int(len(dataset) * (1 - validation_split))
    lengths = [training_length, len(dataset) - training_length]
    train_dataset, validation_dataset = random_split(dataset, lengths)
    return DataLoader(train_dataset, batch_size), DataLoader(validation_dataset, batch_size)


def fit(train_loader, num_dim, val_loader, num_epochs, num_steps, learning_rate):
    """
    The dataset contains 2-D tensor where first dimension runs along batch and 
    second dimension runs along component of an indiviual data point. The y 
    contains the true value of histogram.
    
    """
    
    # defining parameters of the model
    mean_coeff = torch.randn(1, num_dim)
    cov_coeff = torch.randn(num_dim, num_dim)
    a = torch.randn(3, 1)
    b = torch.randn(3, 1)

    # declaring them as learnable parameters
    mean_coeff.requires_grad = True
    cov_coeff.requires_grad = True
    a.requires_grad = True
    b.requires_grad = True

    # setting up the optimizer
    parameters = [mean_coeff, cov_coeff, a, b]
    optimizer = optim.Adam(parameters, learning_rate)
    
    # mean and cov_mat batches calculations
    c_b = (1 / num_steps) * torch.arange(1, 2 * (num_steps + 1), 2)
    c_b.view(-1, 1)
    
    # c_matrix = 
    a_matrix = torch.mm(c_matrix, a)
    b_matrix = torch.mm(c_matrix, b)
    
    mean = torch.mm(a_matrix, mean_coeff)
    # cov = 
    p = torch.distributions.multivariate_normal.MultivariateNormal(mean, cov_mat)
    
    
    for epoch_index in range(num_epochs):
        print("Epoch no. ", epoch_index)
        
        for x, y in train_loader:
            batch_size = x.shape[0]
            x_repeat = ...
            y_pred = (1 / num_steps) * torch.sum(torch.exp(p.log_prob(x_repeat)), dim=1)
            loss = loss_func
            
            
            
        
    
    

num_dim = 2  # number of dimensions
