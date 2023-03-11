import numpy as np
import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange

from nfn.common import WeightSpaceFeatures, NetworkSpec
from nfn.layers.layer_utils import shape_wsfeat_symmetry, unshape_wsfeat_symmetry


class ChannelDropout(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout
        self.matrix_dropout = nn.Dropout2d(dropout)
        self.bias_dropout = nn.Dropout(dropout)

    def forward(self, x: WeightSpaceFeatures) -> WeightSpaceFeatures:
        weights = [self.process_matrix(w) for w in x.weights]
        bias = [self.bias_dropout(b) for b in x.biases]
        return WeightSpaceFeatures(weights, bias)

    def process_matrix(self, mat):
        shape = mat.shape
        is_conv = len(shape) > 4
        if is_conv:
            _, _, _, _, h, w = shape
            mat = rearrange(mat, "b c o i h w -> b (c h w) o i")
        mat = self.matrix_dropout(mat)
        if is_conv:
            mat = rearrange(mat, "b (c h w) o i -> b c o i h w", h=h, w=w)
        return mat


class SimpleLayerNorm(nn.Module):
    def __init__(self, network_spec, channels):
        super().__init__()
        self.network_spec = network_spec
        self.channels = channels
        self.w_norms, self.v_norms = nn.ModuleList(), nn.ModuleList()
        for i in range(len(network_spec)):
            eff_channels = int(channels * np.prod(network_spec.weight_spec[i].shape[2:]))
            self.w_norms.append(ChannelLayerNorm(eff_channels))
            self.v_norms.append(ChannelLayerNorm(channels))

    def forward(self, wsfeat: WeightSpaceFeatures) -> WeightSpaceFeatures:
        wsfeat = shape_wsfeat_symmetry(wsfeat, self.network_spec)
        out_weights, out_biases = [], []
        for i in range(len(self.network_spec)):
            weight, bias = wsfeat[i]
            out_weights.append(self.w_norms[i](weight))
            out_biases.append(self.v_norms[i](bias))
        return unshape_wsfeat_symmetry(WeightSpaceFeatures(out_weights, out_biases), self.network_spec)

    def __repr__(self):
        return f"SimpleLayerNorm(channels={self.channels})"


class ParamLayerNorm(nn.Module):
    def __init__(self, network_spec: NetworkSpec, channels):
        # TODO: This doesn't work for convs yet.
        super().__init__()
        self.n_in, self.n_out = network_spec.get_io()
        self.channels = channels
        for i in range(len(network_spec)):
            if i == 0:
                w_shape = (channels, self.n_in)
                v_shape = (channels,)
            elif i == len(network_spec) - 1:
                w_shape = (self.n_out, channels)
                v_shape = (channels, self.n_out)
            else:
                w_shape = (channels,)
                v_shape = (channels,)
            self.add_module(f"norm{i}_w", nn.LayerNorm(normalized_shape=w_shape))
            self.add_module(f"norm{i}_v", nn.LayerNorm(normalized_shape=v_shape))

    def forward(self, wsfeat: WeightSpaceFeatures) -> WeightSpaceFeatures:
        out_weights, out_biases = [], []
        for i, (weight, bias) in enumerate(wsfeat):
            w_norm = getattr(self, f"norm{i}_w")
            v_norm = getattr(self, f"norm{i}_v")
            if i == 0:
                out_weights.append(w_norm(weight.transpose(-3, -2)).transpose(-3, -2))
                out_biases.append(v_norm(bias.transpose(-1, -2)).transpose(-1, -2))
            elif i == len(wsfeat) - 1:
                out_weights.append(w_norm(weight.transpose(-3, -1)).transpose(-3, -1))
                out_biases.append(v_norm(bias))
            else:
                out_weights.append(w_norm(weight.transpose(-3, -1)).transpose(-3, -1))
                out_biases.append(v_norm(bias.transpose(-1, -2)).transpose(-1, -2))
        return WeightSpaceFeatures(out_weights, out_biases)


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))
        self.channels_last = Rearrange("b c ... -> b ... c")
        self.channels_first = Rearrange("b ... c -> b c ...")

    def forward(self, x):
        # x.shape = (b, c, ...)
        x = self.channels_last(x)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x = (x - mean) / (std + self.eps)
        out = self.channels_first(x * self.gamma + self.beta)
        return out