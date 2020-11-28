import pytest
from pytest import approx
from ip_sgd import logloss, compute_loss
import torch


def test_logloss_trueneg():
    assert logloss(true_label=0., predicted=0., eps=1e-15) == approx(0, abs=1e-15)

def test_logloss_trueneg_torch():
    assert logloss(true_label=torch.tensor(0.), predicted=torch.tensor(0.), eps=1e-15) == approx(0, abs=1e-15)

def test_logloss_truepos():
    assert logloss(true_label=1., predicted=1., eps=1e-15) == approx(0, abs=1e-15)

def test_logloss_midpt():
    assert logloss(true_label=1., predicted=0.5, eps=1e-15) == approx(.693, abs=1e-3)
    assert logloss(true_label=0., predicted=0.5, eps=1e-15) == approx(.693, abs=1e-3)