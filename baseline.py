from argparse import ArgumentParser
import torch
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.bernoulli import Bernoulli
from torch import sigmoid, dot, log
import random


def compute_loss(yhat, y):
    """
    compute logistic loss function
    """
    if (yhat == 0 and y == 0) or (yhat == 1 and y == 1):
        return 0
    return -1 * (y * log(yhat) + (1 - y) * log(1 - yhat))


def main(args):
    # fix random seed
    torch.manual_seed(args.seed)

    # set up tensorboard logging
    label = 'baseline'
    if args.ip:
        label = 'inner-product-sgd'
    writer = SummaryWriter(f'runs/{label}')

    # dataset parameters
    input_dim = 50
    A = torch.rand(input_dim, input_dim)
    covariance_matrix = torch.matmul(A.t(), A)
    m = MultivariateNormal(torch.zeros(input_dim), covariance_matrix)
    b = torch.randn(input_dim)

    # create dataset
    inputs = [m.sample() for _ in range(args.num_examples)]
    data = [(x, Bernoulli(sigmoid(dot(b, x))).sample()) for x in inputs]

    # train/val split
    split_index = int(len(data) * 0.8)
    train = data[:split_index]
    val = data[split_index:]
    print(f'{len(train)} train examples, {len(val)} val examples')

    # training initialization
    bhat = torch.rand(input_dim)
    step_size = 0.001

    # training loop
    for epoch in range(args.epochs):
        if args.ip:
            train.sort(key=lambda example: dot(example[0], bhat))
        else:
            random.shuffle(train)
        print(f"----- epoch {epoch} -----")
        for i, (x, y) in enumerate(train):
            # predict and compute loss
            yhat = sigmoid(dot(bhat, x))
            loss = compute_loss(yhat, y)
            writer.add_scalar('train_loss', loss, epoch * len(data) + i)

            # compute gradient and update b_hat
            gradient = x * (yhat - y)
            bhat -= step_size * gradient

            # compute and log || b_hat - b ||
            b_error = np.linalg.norm(bhat - b, 2)
            writer.add_scalar('b_error', b_error, epoch * len(data) + i)

        # compute and report avg loss on validation set
        val_loss = 0
        for i, (x, y) in enumerate(val):
            yhat = sigmoid(dot(bhat, x))
            loss = compute_loss(yhat, y)
            val_loss += loss
        val_loss_avg = val_loss / len(val)
        writer.add_scalar('avg_val_loss', val_loss_avg, epoch)
        print(f'val loss: {val_loss_avg}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--ip',
        action='store_true',
        help='if set, bias SGD by inner product <x_i, b>',
    )
    parser.add_argument(
        '--num-epochs', '-e',
        type=int,
        default=40,
    )
    parser.add_argument(
        '--num-examples', '-n',
        type=int,
        default=10000
    )
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=42
    )
    args = parser.parse_args()
    main(args)
