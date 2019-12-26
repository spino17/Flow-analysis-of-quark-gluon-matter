import torch
from torch import nn
import os
import pandas
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import torch.optim as optim
import matplotlib.pyplot as plt


def mean_squared_error(y_pred, y_true):
    """
    Used in the original paper
    
    """

    nume = (y_pred - y_true) ** 2
    # print(nume)
    # deno = y_true
    return torch.sum(nume) / torch.sum(y_true)


def cross_entropy_loss(y_pred, y_true):
    return -1 * torch.sum(y_true * torch.log(y_pred))


def data_preprocessing(X, y, batch_size, validation_split=0.2):
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()
    dataset = TensorDataset(X, y)
    batch_size = batch_size
    training_length = int(len(dataset) * (1 - validation_split))
    lengths = [training_length, len(dataset) - training_length]
    train_dataset, validation_dataset = random_split(dataset, lengths)
    return (
        DataLoader(train_dataset, batch_size),
        DataLoader(validation_dataset, batch_size),
    )


def fit(train_loader, val_loader, num_dim, num_epochs, num_steps, learning_rate):
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
    lower_coeff = torch.randn(
        int(num_dim * (num_dim + 1) / 2)
    )  # vectorized lower triangular matrix
    a = torch.randn(3, 1)
    b = torch.randn(3, 1)

    # declaring them as learnable parameters
    mean_coeff.requires_grad = True
    lower_coeff.requires_grad = True
    a.requires_grad = True
    b.requires_grad = True

    # setting up the optimizer
    parameters = [mean_coeff, lower_coeff, a, b]
    optimizer = optim.Adam(parameters, learning_rate)

    # mean and cov_mat batches calculations - mid point riemannian sum
    c_b = (1 / (2 * num_steps)) * torch.arange(1, 2 * num_steps + 1, 2)
    c_b = c_b.view(-1, 1)
    c_matrix = torch.cat((c_b, c_b ** 2, c_b ** 3), 1)
    a_matrix = torch.mm(c_matrix, a)
    b_matrix = torch.mm(c_matrix, b)

    # forming lower triangular matrix
    lower_indices = torch.tril_indices(num_dim, num_dim)
    lower_matrix = torch.zeros((num_dim, num_dim))
    lower_matrix[lower_indices[0], lower_indices[1]] = lower_coeff

    # defining batch multivariate normal distribution
    mean = torch.mm(
        a_matrix, mean_coeff
    )  # batch mean along all values of c_b step varying from 0 to 1
    cov_mat_l = (b_matrix * lower_matrix.view(1, -1)).view(
        -1, num_dim, num_dim
    )  # batch cholesky lower triangular matrix
    cov_mat = torch.matmul(
        cov_mat_l, torch.transpose(cov_mat_l, 1, 2)
    )  # cholesky decomposition
    p = torch.distributions.multivariate_normal.MultivariateNormal(
        mean, cov_mat
    )  # batch normal distribution

    print(mean[:10])
    print(cov_mat[:10])
    # training loop
    train_losses, val_losses, epochs = [], [], []
    train_len = len(train_loader)
    val_len = len(val_loader)
    print("Training starting !")
    for epoch_index in range(num_epochs):
        print("Epoch no. ", epoch_index)
        train_loss = 0
        val_loss = 0
        batch_count = 0
        for x, y in train_loader:
            N_total = torch.tensor(torch.sum(y))
            optimizer.zero_grad()
            batch_size = x.shape[0]
            x_repeat = (
                x.repeat(1, num_steps).view(-1, num_dim).view(batch_size, -1, num_dim)
            )
            # print(p.log_prob(x_repeat))
            y_pred = (1 / num_steps) * torch.sum(
                torch.exp(torch.add(p.log_prob(x_repeat), torch.log(N_total))), dim=1
            )
            loss = mean_squared_error(
                y_pred, y
            )  # loss function between predicted and true values
            loss.backward(retain_graph=True)  # calculate gradients
            optimizer.step()  # take a small step in the direction of gradient
            train_loss += loss.item()
            batch_count += 1
            break
        else:
            with torch.no_grad():
                # scope of no gradient calculations
                for x, y in val_loader:
                    batch_size = x.shape[0]
                    x_repeat = (
                        x.repeat(1, num_steps)
                        .view(-1, num_dim)
                        .view(batch_size, -1, num_dim)
                    )
                    y_pred = (1 / num_steps) * torch.sum(
                        torch.exp(p.log_prob(x_repeat)), dim=1
                    )
                    loss = mean_squared_error(y_pred, y)
                    val_loss += loss.item()

            train_losses.append(train_loss / train_len)
            val_losses.append(val_loss / val_len)
            epochs.append(epoch_index)
            print(
                "Train loss: %.2f - Val loss: %.2f"
                % (train_loss / train_len, val_loss / val_len)
            )

    print("Training finished !")

    plt.plot(epochs, train_losses, color="red")
    plt.plot(epochs, val_losses, color="blue")
    plt.show()

    return mean_coeff, lower_coeff, a, b


def main():
    # load the dataset
    X = np.load("data/x.npy")
    y = np.load("data/y.npy").reshape(-1, 1)

    # defining hyperparameters
    batch_size = 100
    validation_split = 0.2
    num_dim = 2
    num_epochs = 50
    num_steps = 500
    learning_rate = 0.1

    # prepare dataloader for optimization
    train_loader, val_loader = data_preprocessing(X, y, batch_size, validation_split)

    # training the model
    mean_coeff, cov_coeff, a, b = fit(
        train_loader, val_loader, num_dim, num_epochs, num_steps, learning_rate
    )


if __name__ == "__main__":
    main()  # calling the main function
