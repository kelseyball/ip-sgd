import torch
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.bernoulli import Bernoulli
from torch import sigmoid, dot, log
import random

input_dim = 50
num_examples = 10000
step_size = 0.001
num_epochs = 20

writer = SummaryWriter('runs/baseline')

A = torch.rand(input_dim, input_dim)
covariance_matrix = torch.matmul(A.t(), A)
m = MultivariateNormal(torch.zeros(input_dim), covariance_matrix)
b = torch.randn(input_dim)

inputs = [m.sample() for _ in range(num_examples)]
data = [(x, Bernoulli(sigmoid(dot(b, x))).sample()) for x in inputs]

split_index = int(len(data) * 0.8)
print(split_index)
train = data[:split_index]
val = data[split_index:]
print(f'{len(train)} train examples, {len(val)} val examples')

bhat = torch.rand(input_dim)

yhat_counter = 0
for epoch in range(num_epochs):
    random.shuffle(train)
    print(f"----- epoch {epoch} -----")
    for i, (x, y) in enumerate(train):
        yhat = sigmoid(dot(bhat, x))

        # avoid NaN/inf
        if yhat == 0 and y == 0:
            loss = 0
            yhat_counter += 1
        elif yhat == 1 and y == 1:
            loss = 0
            yhat_counter += 1
        else:
            loss = -1 * (y * log(yhat) + (1 - y) * log(1 - yhat))
            writer.add_scalar('train_loss', loss, epoch * len(data) + i)

        # compute/update the gradient for each b_i
        gradient = x * (yhat - y)
        bhat -= step_size * gradient

        b_error = np.linalg.norm(bhat - b, 2)
        writer.add_scalar('b_error', b_error, epoch * len(data) + i)

    val_loss = 0
    for i, (x, y) in enumerate(val):
        yhat = sigmoid(dot(bhat, x))
        # avoid NaN/inf
        if (yhat == 0 and y == 0) or (yhat == 1 and y == 1):
            loss = 0
        else:
            loss = -1 * (y * log(yhat) + (1 - y) * log(1 - yhat))
        val_loss += loss

    val_loss_avg = val_loss / len(val)
    writer.add_scalar('avg_val_loss', val_loss / len(val), epoch)
    b_error = np.linalg.norm(bhat - b, 2)
    writer.add_scalar('b_error', b_error, epoch * len(data) + i)
    print(f'val loss: {val_loss_avg}')
