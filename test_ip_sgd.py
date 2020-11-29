import pytest
from pytest import approx
from ip_sgd import logloss, normalize, max_ip_negative
import torch

input_dim = 2

@pytest.fixture
def bhat():
    return torch.ones(input_dim)

@pytest.fixture
def train():
    return [
        (torch.full((input_dim,), fill_value=1.0), torch.tensor(1.0)),
        (torch.full((input_dim,), fill_value=-1.0), torch.tensor(0.0)),
        (torch.full((input_dim,), fill_value=2.0), torch.tensor(0.0)), # this is the "informative" negative
        (torch.full((input_dim,), fill_value=-2.0), torch.tensor(0.0)),
        (torch.full((input_dim,), fill_value=3.0), torch.tensor(1.0)),
        (torch.full((input_dim,), fill_value=-3.0), torch.tensor(1.0)),
    ]

def test_logloss_trueneg():
    assert logloss(true_label=0., predicted=0., eps=1e-15) == approx(0, abs=1e-15)

def test_logloss_trueneg_torch():
    assert logloss(true_label=torch.tensor(0.), predicted=torch.tensor(0.), eps=1e-15) == approx(0, abs=1e-15)

def test_logloss_truepos():
    assert logloss(true_label=1., predicted=1., eps=1e-15) == approx(0, abs=1e-15)

def test_logloss_midpt():
    assert logloss(true_label=1., predicted=0.5, eps=1e-15) == approx(.693, abs=1e-3)
    assert logloss(true_label=0., predicted=0.5, eps=1e-15) == approx(.693, abs=1e-3)

def test_normalize_l2():
    b = torch.randn(50)
    assert torch.norm(b, p=2).item() != 1.
    assert torch.norm(normalize(b), p=2).item() == approx(1, abs=1e-5)

def test_normalize_l1():
    b = torch.randn(50)
    assert torch.norm(b, p=1).item() != 1.
    assert torch.norm(normalize(b, p=1), p=1).item() == approx(1, abs=1e-5)

def test_max_ip_negative(train, bhat):
    max_negative_example = max_ip_negative(train=train, bhat=bhat)
    assert torch.all(torch.eq(max_negative_example[0], torch.full((input_dim,), fill_value=2.0)))
