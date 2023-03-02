from collections import OrderedDict
from torch import nn
import torch


def make_cnn():
    """Define a small CNN whose weight space we will process with an NFN."""
    return nn.Sequential(
        nn.Conv2d(1, 32, 3), nn.ReLU(),
        nn.Conv2d(32, 32, 3), nn.ReLU(),
        nn.Conv2d(32, 32, 3), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 10)
    )


def check_same(cnn1, cnn2):
    inp = torch.randn(20, 1, 28, 28)
    out1 = cnn1(inp)
    out2 = cnn2(inp)
    return torch.allclose(out1, out2)


def sample_perm(state_dict: OrderedDict):
    prev_perm = None
    i = 0
    sd_list = list(state_dict.items())
    permuted_sd = OrderedDict()
    while i < len(state_dict):
        weight_key, weight = sd_list[i]
        bias_key, bias = sd_list[i + 1]
        if prev_perm is not None:
            weight = weight[:, prev_perm]
        if i + 2 < len(state_dict):
            this_perm = torch.randperm(weight.shape[0])
            weight = weight[this_perm]
            bias = bias[this_perm]
            prev_perm = this_perm
        permuted_sd[weight_key] = weight
        permuted_sd[bias_key] = bias
        i += 2
    return permuted_sd


def check_perm_symmetry():
    # Sanity check that we are permuting CNN channels properly.
    cnn1, cnn2 = make_cnn(), make_cnn()
    cnn2.load_state_dict(sample_perm(cnn1.state_dict()))
    return check_same(cnn1, cnn2)
