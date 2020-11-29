from typing import List, Tuple
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


def random_positive(train: List[Tuple[torch.tensor, torch.tensor]]):
    """
    return random positive from dataset
    """
    positives = [e for e in train if e[1].item() == 1.0]
    return random.choice(positives)


def max_ip_negative(train: List[Tuple[torch.tensor, torch.tensor]], bhat: torch.tensor):
    """
    return negative with max inner product with bhat
    """
    negatives_with_ip = [(e, dot(e[0], bhat)) for e in train if e[1].item() == 0.0]
    negative, _ = max(negatives_with_ip, key=lambda x: x[1])
    return negative


def normalize(t: torch.tensor, p=2) -> torch.tensor:
    """
    make unit norm w.r.t l-p norm
    """
    return t.div(torch.norm(t, p=p))


def logloss(predicted, true_label, eps=1e-15) -> float:
    p = np.clip(predicted, eps, 1 - eps)
    if true_label == 1:
        return -np.log(p)
    else:
        return -np.log(1 - p)


def generate_data(m: MultivariateNormal, b: torch.tensor, num_examples: int, linearly_separable: bool, positives: float) -> List[Tuple[torch.tensor, torch.tensor]]:
    """
    create dataset with given parameters
    """
    inputs = [m.sample() for _ in range(num_examples)]
    data = [(x, Bernoulli(sigmoid(dot(b, x))).sample()) for x in inputs]
    if linearly_separable:
        data = [(x, torch.tensor(1.0) if sigmoid(dot(b, x)) > 0.5 else torch.tensor(0)) for x in inputs]
    if positives < 1.0:
        data = [(x, y) for (x, y) in data if (y.item() == 0.0) or (y.item() == 1.0 and random.random() < args.positives)]
    return data


def describe_data(data: List[Tuple[torch.tensor, torch.tensor]], writer: SummaryWriter):
    """
    log/report count, ratio of positives and negatives
    """
    num_positives = sum([1 for e in data if e[1].item() == 1.0])
    num_negatives = sum([1 for e in data if e[1].item() == 0.0])
    writer.add_text('num_positives', str(num_positives))
    writer.add_text('num_negatives', str(num_negatives))


def main(args):
    if args.seed:
        # fix random seed
        torch.manual_seed(args.seed)

    # set up tensorboard logging
    algo = 'ip-sgd' if args.ip else 'vanilla-sgd'
    LS = '_LS' if args.ls else ''
    label = f'{algo}{LS}_n{args.num_examples}_e{args.num_epochs}_ss{args.step_size}_p{args.positives}'
    if args.label:
        label += f'_{args.label}'
    writer = SummaryWriter(f'{args.folder}/{label}')
    writer.add_text('args', str(args))

    # create dataset
    input_dim = args.dim
    m = MultivariateNormal(torch.zeros(input_dim), torch.eye(input_dim))
    b = torch.randn(input_dim)
    data = generate_data(m, b, args.num_examples, args.ls, args.positives)
    describe_data(data, writer)

    # train/val split
    split_index = int(len(data) * 0.8)
    train = data[:split_index]
    val = data[split_index:]
    writer.add_text('num train examples', str(len(train)))
    writer.add_text('num val examples', str(len(val)))
    _, y_true_val = tuple(zip(*val))

    # training initialization
    bhat = torch.randn(input_dim)
    step_size = args.step_size

    # training loop
    for epoch in range(args.num_epochs):
        random.shuffle(train)
        print(f"----- epoch {epoch} -----")
        for i in trange(len(train)):
            x, y = None, None
            if args.ip:
                if random.random() < 0.5:
                    (x, y) = random_positive(train)
                else:
                    (x, y) = max_ip_negative(train, bhat)
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
        '--dim', '-d',
        type=int,
        default=50,
        help='dimension of data',
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
        default=1,
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
    parser.add_argument(
        '--ls',
        action='store_true',
        help='if set, make data linearly separable'
    )
    args = parser.parse_args()
    main(args)
