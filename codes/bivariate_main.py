import torch
from torch import nn
import os
import pandas
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import torch.optim as optim


def mean_squared_error(y_pred, y_true):
    """
    Used in the original paper
    
    """   
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
    return (
        DataLoader(train_dataset, batch_size),
        DataLoader(validation_dataset, batch_size),
    )


def fit(train_loader, num_dim, val_loader, num_epochs, num_steps, learning_rate):
    """
    The dataset contains 2-D tensor where first dimension runs along batch and 
    second dimension runs along component of an indiviual data point. The y 
    contains the true value of histogram.
    
    The optimization algorithm used here is Stochastic Gradient Descent (SGD)
    and data is batched according to the batch_size. The gradients calculated 
    at each epoch are thus a stochastic estimate of the gradient over the com-
    plete dataset.
    
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

    # mean and cov_mat batches calculations - mid point riemannian sum
    c_b = (1 / num_steps) * torch.arange(1, 2 * (num_steps + 1), 2)
    c_b = c_b.view(-1, 1)
    c_matrix = torch.cat((c_b, c_b ** 2, c_b ** 3), 1)
    a_matrix = torch.mm(c_matrix, a)
    b_matrix = torch.mm(c_matrix, b)

    # defining batch multivariate normal distribution
    mean = torch.mm(
        a_matrix, mean_coeff
    )  # batch mean along all values of c_b step varying from 0 to 1
    cov_mat = (b_matrix * cov_coeff.view(1, -1)).view(
        -1, num_dim, num_dim
    )  # batch covariance matrix
    p = torch.distributions.multivariate_normal.MultivariateNormal(
        mean, cov_mat
    )  # batch normal distribution

    # training loop
    train_losses, val_losses, epochs = [], [], []
    train_len = len(train_loader)
    val_len = len(val_loader)
    for epoch_index in range(num_epochs):
        print("Epoch no. ", epoch_index)

        for x, y in train_loader:
            optimizer.zero_grad()
            batch_size = x.shape[0]
            # x_repeat = ...
            y_pred = (1 / num_steps) * torch.sum(torch.exp(p.log_prob(x_repeat)), dim=1)
            loss = mean_squared_error(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_losses.append(train_loss / train_len)
    
    print("training finished")


num_dim = 2  # number of dimensions
