import math
import torch
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.bernoulli import Bernoulli
from torch import sigmoid, dot, log
from torch import optim
from torch.nn import BCELoss

input_dim = 50
num_examples = 1000
step_size = 0.01
num_epochs = 10

m = MultivariateNormal(torch.zeros(input_dim), torch.eye(input_dim))
b = torch.rand(input_dim)
print("b.shape: ", b.shape)

inputs = [m.sample() for _ in range(num_examples)]
print("x.shape: ", inputs[0].shape)

data = [(x, Bernoulli(sigmoid(dot(b, x))).sample()) for x in inputs]
print("y[:5]: ")
print([(d[1], d[1].type()) for d in data[:5]])

bhat = torch.rand(input_dim)
for _ in range(num_epochs):
	for (x, y) in data:
	  yhat = sigmoid(dot(bhat, x))
	  print("y, yhat: ", y, yhat)
	  loss = -1 * (y * log(yhat)) + (1 - y) * log(1 - yhat)
	  # print("loss: ", loss)

	  # compute/update the gradient for each w_i
	  for i in range(input_dim):
	    dbhat_i = x[i] * (yhat - y)
	    bhat[i] = bhat[i] - step_size * dbhat_i

	  b_errors = [bhat[i] - b[i] for i in range(input_dim)]
	  b_error = np.linalg.norm(b_errors, 2)
	  print("b_error: ", b_error)

