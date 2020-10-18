import torch
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.bernoulli import Bernoulli
from torch import sigmoid, dot, log
import random

input_dim = 50
num_examples = 1000
step_size = 0.001
num_epochs = 10

writer = SummaryWriter('runs/baseline')

# TODO: generate random covariance matrix
A = torch.randn(input_dim, input_dim)
covariance_matrix = torch.matmul(A.t(), A)
m = MultivariateNormal(torch.zeros(input_dim), covariance_matrix)
b = torch.randn(input_dim)

inputs = [m.sample() for _ in range(num_examples)]
data = [(x, Bernoulli(sigmoid(dot(b, x))).sample()) for x in inputs]

bhat = torch.rand(input_dim)

for epoch in range(num_epochs):
    random.shuffle(data)
    print(f"----- epoch {epoch} -----")
    for i, (x, y) in enumerate(data):
        yhat = sigmoid(dot(bhat, x))
        # print("y, yhat: ", y, yhat)
        loss = -1 * (y * log(yhat)) + (1 - y) * log(1 - yhat)
        writer.add_scalar('train_loss', loss, epoch * len(data) + i)

        # compute/update the gradient for each b_i
        gradient = x * (yhat - y)
        bhat -= step_size * gradient

        b_error = np.linalg.norm(bhat - b, 2)
        writer.add_scalar('b_error', b_error, epoch * len(data) + i)
