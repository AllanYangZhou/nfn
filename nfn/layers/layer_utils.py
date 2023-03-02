import math
from torch import nn
from nfn.common import WeightSpaceFeatures
from einops import rearrange


def set_init_(*layers):
    in_chan = 0
    for layer in layers:
        if isinstance(layer, (nn.Conv2d, nn.Conv1d)):
            in_chan += layer.in_channels
        elif isinstance(layer, nn.Linear):
            in_chan += layer.in_features
        else:
            raise NotImplementedError(f"Unknown layer type {type(layer)}")
    bd = math.sqrt(1 / in_chan)
    for layer in layers:
        nn.init.uniform_(layer.weight, -bd, bd)
        if layer.bias is not None:
            nn.init.uniform_(layer.bias, -bd, bd)


def shape_wsfeat_symmetry(params, network_spec):
    """Reshape so last 2 dims have symmetry, channel dims have all nonsymmetry.
    E.g., for conv weights we reshape (B, C, out, in, h, w) -> (B, C * h * w, out, in)
    """
    weights, bias = params.weights, params.biases
    reshaped_weights = []
    for weight, weight_spec in zip(weights, network_spec.weight_spec):
        if len(weight_spec.shape) == 2:  # mlp weight matrix:
            reshaped_weights.append(weight)
        else:
            reshaped_weights.append(rearrange(weight, "b c o i h w -> b (c h w) o i"))
    return WeightSpaceFeatures(reshaped_weights, bias)


def unshape_wsfeat_symmetry(params, network_spec):
    """Reverse shape_params_symmetry"""
    weights, bias = params.weights, params.biases
    unreshaped_weights = []
    for weight, weight_spec in zip(weights, network_spec.weight_spec):
        if len(weight_spec.shape) == 2:  # mlp weight matrix:
            unreshaped_weights.append(weight)
        else:
            _, _, h, w = weight_spec.shape
            unreshaped_weights.append(rearrange(weight, "b (c h w) o i -> b c o i h w", h=h, w=w))
    return WeightSpaceFeatures(unreshaped_weights, bias)
