import numpy as np
from torch import nn
from einops import rearrange

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
        for i in range(len(network_spec)):
            eff_channels = int(channels * np.prod(network_spec.weight_spec[i].shape[2:]))
            self.add_module(f"norm{i}_w", nn.LayerNorm(normalized_shape=(eff_channels,)))
            self.add_module(f"norm{i}_v", nn.LayerNorm(normalized_shape=(channels,)))

    def forward(self, wsfeat: WeightSpaceFeatures) -> WeightSpaceFeatures:
        wsfeat = shape_wsfeat_symmetry(wsfeat, self.network_spec)
        in_weights, in_biases = [], []
        for weight, bias in wsfeat:
            in_weights.append(weight.transpose(-3, -1))
            in_biases.append(bias.transpose(-1, -2))
        out_weights, out_biases = [], []
        for i, (weight, bias) in enumerate(zip(in_weights, in_biases)):
            w_norm = getattr(self, f"norm{i}_w")
            v_norm = getattr(self, f"norm{i}_v")
            out_weights.append(w_norm(weight))
            out_biases.append(v_norm(bias))
        out_weights = [w.transpose(-3, -1) for w in out_weights]
        out_biases = [b.transpose(-1, -2) for b in out_biases]
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
