from argparse import ArgumentParser
from tqdm import trange
import torch
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.bernoulli import Bernoulli
from torch import sigmoid, dot, log
import random


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
    input_dim = 50
    m = MultivariateNormal(torch.zeros(input_dim), torch.eye(input_dim))
    b = torch.rand(input_dim)
    # b = b.div(torch.norm(b, p=2))

    # create dataset
    inputs = [m.sample() for _ in range(args.num_examples)]
    # data = [(x, Bernoulli(sigmoid(dot(b, x))).sample()) for x in inputs]

    # try making data linearly separable
    data = [(x, torch.tensor(1.0) if sigmoid(dot(b, x)) > 0.5 else torch.tensor(0)) for x in inputs]

    # throwaway 80% of positives
    # data = [(x, y) for (x, y) in data if (y.item() == 0.0) or (y.item() == 1.0 and random.random() < args.positives)]

    data_rebalanced = []
    for (x, y) in data:
        if y.item() == 0.0:
            data_rebalanced.append((x, y))
        else:
            if random.random() < args.positives:
                data_rebalanced.append((x,y))
            else:
                # replace with negative
                while True:
                    new_x = m.sample()
                    new_y = sigmoid(dot(b, new_x))
                    if new_y < 0.5:
                        data_rebalanced.append((new_x, torch.tensor(0)))
                        break
    data = data_rebalanced


    # report count, ratio of positives and negatives
    positives = [e for e in data if e[1].item() == 1.0]
    negatives = [e for e in data if e[1].item() == 0.0]
    print(f'{len(positives)} positives, {len(negatives)} negatives')
    print(f'{len(positives)/float(len(data)):.2f}% positives, {len(negatives)/float(len(data)):.2f}% negatives')

    # train/val split
    split_index = int(len(data) * 0.8)
    train = data[:split_index]
    val = data[split_index:]
    print(f'{len(train)} train examples, {len(val)} val examples')

    # training initialization
    bhat = torch.rand(input_dim)
    # bhat = bhat.div(torch.norm(bhat, p=2))
    step_size = args.step_size

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

            # compute gradient and update b_hat
            gradient = x * (yhat - y)
            bhat -= step_size * gradient

            # compute and log || b_hat - b ||
            unnormalized_b_error = np.linalg.norm(bhat - b, 2)
            # b_error = np.linalg.norm(bhat - b, 2) / np.linalg.norm(b, 2)
            # writer.add_scalar('b_error', b_error, epoch * len(data) + i)
            # writer.add_scalar('log_b_error', np.log(b_error), epoch * len(data) + i)
            # writer.add_scalar('unnormalized_b_error', unnormalized_b_error, epoch * len(data) + i)

        # compute and report avg loss on validation set
        val_loss = 0
        for i, (x, y) in enumerate(val):
            yhat = sigmoid(dot(bhat, x))
            loss = compute_loss(yhat, y)
            val_loss += loss
        val_loss_avg = val_loss / len(val)

        b_error = np.linalg.norm(bhat - b, 2) / np.linalg.norm(b, 2)
        writer.add_scalar('b_error', b_error, epoch)
        writer.add_scalar('avg_val_loss', val_loss_avg, epoch)
        print(f'val loss: {val_loss_avg}')


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
        default=40,
    )
    parser.add_argument(
        '--num-examples', '-n',
        type=int,
        default=1000
    )
    parser.add_argument(
        '--seed', '-r',
        type=int,
        default=42
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
        default=0.001,
    )
    parser.add_argument(
        '--positives', '-p',
        type=float,
        default=0.2,
        help='percent positives to keep'
    )
    args = parser.parse_args()
    main(args)
