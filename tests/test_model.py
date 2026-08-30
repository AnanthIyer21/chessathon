import sys

import torch

sys.path.insert(0, ".")  # repo root, for train package
from train.model import PolicyValueNet


def test_forward_shapes_and_ranges():
    net = PolicyValueNet(channels=32, blocks=2)
    x = torch.randn(4, 19, 8, 8)
    policy, value = net(x)
    assert policy.shape == (4, 4672)
    assert value.shape == (4,)
    assert value.abs().max() <= 1.0


def test_param_count_default():
    net = PolicyValueNet()
    n = sum(p.numel() for p in net.parameters())
    assert 1_000_000 < n < 5_000_000
