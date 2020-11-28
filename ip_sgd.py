from argparse import ArgumentParser
from tqdm import trange
import torch
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.bernoulli import Bernoulli
from torch import sigmoid, dot, log
import random

from sklearn.metrics import f1_score


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


def logloss(predicted, true_label, eps=1e-15):
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
    algo = 'ip-sgd' if args.ip else 'vanilla-sgd'
    label = f'{algo}_n{args.num_examples}_ss{args.step_size}_p{args.positives}'
    if args.label:
        label += f'_{args.label}'
    writer = SummaryWriter(f'{args.folder}/{label}')
    writer.add_text('args', str(args))

    # dataset parameters
    input_dim = 50
    m = MultivariateNormal(torch.zeros(input_dim), torch.eye(input_dim))
    b = torch.rand(input_dim)

    # create dataset
    inputs = [m.sample() for _ in range(args.num_examples)]
    # data = [(x, Bernoulli(sigmoid(dot(b, x))).sample()) for x in inputs]

    # try making data linearly separable
    data = [(x, torch.tensor(1.0) if sigmoid(dot(b, x)) > 0.5 else torch.tensor(0)) for x in inputs]

    if args.positives < 1.0:
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
            loss = logloss(yhat, y)
            writer.add_scalar('train_loss', loss, epoch * len(data) + i)

            # compute gradient and update b_hat
            gradient = x * (yhat - y)
            bhat -= step_size * gradient

        # compute and report avg loss on validation set
        val_loss = 0
        y_pred_val = []
        for i, (x, y) in enumerate(val):
            yhat = sigmoid(dot(bhat, x))
            y_pred_val.append(1 if yhat > 0.5 else 0)
            loss = logloss(yhat, y)
            val_loss += loss

        val_loss_avg = val_loss / len(val)
        f1 = f1_score(y_true_val, y_pred_val)
        b_error = np.linalg.norm(bhat - b, 2) / np.linalg.norm(b, 2)
        writer.add_scalar('avg_val_loss', val_loss_avg, epoch)
        writer.add_scalar('f1', f1, epoch)
        writer.add_scalar('b_error', b_error, epoch)
        print(f'val loss: {val_loss_avg}')
        print(f'f1 score: {f1}')


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
        default=1.0,
        help='percent positives to keep'
    )
    args = parser.parse_args()
    main(args)
