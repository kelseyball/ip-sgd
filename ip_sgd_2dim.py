from argparse import ArgumentParser
from tqdm import trange
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.bernoulli import Bernoulli
from torch import sigmoid, dot, log
import random

from sklearn.metrics import f1_score

def plot_decision_boundary(bhat, b):
    """
    Plot true and estimated decision boundary
    """
    x = np.linspace(-5,5,100)
    y = -1 * (bhat[0] / bhat[1]) * x
    plt.plot(x, y, 'c')
    y = -1 * (b[0] / b[1]) * x
    plt.plot(x, y, 'm')
    plt.show()

def plot_point(example):
    """
    plot x in blue if positive label, else red
    """
    ((x1, x2), y) = example
    plt.plot(x1, x2, 'bo' if y.item() == 1.0 else 'ro')


def compute_loss(yhat, y):
    """
    compute logistic loss
    """
    if (yhat == 0 and y == 0) or (yhat == 1 and y == 1):
        return 0
    return -1 * (y * log(yhat) + (1 - y) * log(1 - yhat))


def sort_negatives(train, bhat):
    """
    interleave shuffled positives with negatives sorted by <x_i, b_hat>
    """
    positives = [e for e in train if e[1].item() == 1.0]
    negatives = [e for e in train if e[1].item() == 0.0]
    random.shuffle(positives)
    negatives.sort(
        key=lambda example: dot(example[0], bhat),
        reverse=True,
    )
    return [
        example
        for pair in zip(positives, negatives)
        for example in pair
    ]


def logloss(true_label, predicted, eps=1e-15):
    p = np.clip(predicted, eps, 1 - eps)
    if true_label == 1:
        return -np.log(p)
    else:
        return -np.log(1 - p)


def main(args):
    if args.seed:
        # fix random seed
        torch.manual_seed(args.seed)

    # set up tensorboard logging
    label = 'vanilla-sgd'
    if args.ip:
        label = 'inner-product-sgd'
    if args.label:
        label = args.label
    writer = SummaryWriter(f'{args.folder}/{label}')

    # dataset parameters
    input_dim = 2 
    m = MultivariateNormal(torch.zeros(input_dim), args.covariance * torch.eye(input_dim))
    b = torch.rand(input_dim)

    # create dataset
    inputs = [m.sample() for _ in range(args.num_examples)]
    data = [(x, Bernoulli(sigmoid(dot(b, x))).sample()) for x in inputs]

    # try making data linearly separable
    # data = [(x, torch.tensor(1.0) if sigmoid(dot(b, x)) > 0.5 else torch.tensor(0)) for x in inputs]

    if args.positives < 1:
        # throwaway positives to skew ratio
        data = [(x, y) for (x, y) in data if (y.item() == 0.0) or (y.item() == 1.0 and random.random() < args.positives)]

    # report count, ratio of positives and negatives
    num_positives = sum([1 for e in data if e[1].item() == 1.0])
    num_negatives = sum([1 for e in data if e[1].item() == 0.0])
    print(f'{num_positives} positives, {num_negatives} negatives')
    print(f'{num_positives/float(len(data)):.2f}% positives, {num_negatives/float(len(data)):.2f}% negatives')

    # train/val split
    split_index = int(len(data) * 0.8)
    train = data[:split_index]
    val = data[split_index:]
    print(f'{len(train)} train examples, {len(val)} val examples')
    _, y_true_val = tuple(zip(*val))

    # training initialization
    bhat = torch.rand(input_dim)
    step_size = args.step_size

    # plot data and initial decision boundary
    for e in data:
        plot_point(e)
    plot_decision_boundary(bhat, b)

    # training loop
    for epoch in range(args.num_epochs):
        random.shuffle(train)
        print(f"----- epoch {epoch} -----")
        for i in trange(len(train)):
            x, y = None, None
            if args.ip:
                train = sort_negatives(train, bhat)
                (x, y) = random.choice([train[0], train[1]])
            else:
                (x, y) = train[i]

            # predict and compute loss
            yhat = sigmoid(dot(bhat, x))
            loss = compute_loss(yhat, y)
            writer.add_scalar('train_loss', loss, epoch * len(data) + i)
            plot_point((x, y))

            # compute gradient and update b_hat
            gradient = x * (yhat - y)
            bhat -= step_size * gradient

        plot_decision_boundary(bhat, b)

        # compute and report avg loss on validation set
        val_loss = 0
        y_pred_val = []
        for i, (x, y) in enumerate(val):
            yhat = sigmoid(dot(bhat, x))
            y_pred_val.append(1 if yhat > 0.5 else 0)
            loss = compute_loss(yhat, y)
            val_loss += loss
            plot_point((x, y))

        val_loss_avg = val_loss / len(val)
        f1 = f1_score(y_true_val, y_pred_val)
        b_error = np.linalg.norm(bhat - b, 2) / np.linalg.norm(b, 2)
        writer.add_scalar('avg_val_loss', val_loss_avg, epoch)
        writer.add_scalar('f1', f1, epoch)
        writer.add_scalar('b_error', b_error, epoch)
        print(f'val loss: {val_loss_avg}')
        print(f'f1 score: {f1}')
        print(f'b error: {b_error}')
        plot_decision_boundary(bhat, b)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--ip',
        action='store_true',
        help='if set, sort examples by inner product <x_i, b>',
    )
    parser.add_argument(
        '--num-epochs', '-e',
        type=int,
        default=5,
    )
    parser.add_argument(
        '--num-examples', '-n',
        type=int,
        default=100
    )
    parser.add_argument(
        '--seed', '-r',
        type=int,
    )
    parser.add_argument(
        '--label', '-l',
        help='tensorboard run label',
    )
    parser.add_argument(
        '--folder', '-f',
        help='tensorboard run folder',
        default='runs'
    )
    parser.add_argument(
        '--step-size', '-s',
        type=float,
        default=0.01,
    )
    parser.add_argument(
        '--positives', '-p',
        type=float,
        default=1.0,
        help='percent positives to keep'
    )
    parser.add_argument(
        '--covariance', '-c',
        type=float,
        default=1.0,
        help='factor to scale identity covariance matrix by'
    )
    args = parser.parse_args()
    main(args)
