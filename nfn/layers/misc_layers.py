import numpy as np
import torch
from torch import nn
from nfn.layers.layer_utils import shape_wsfeat_symmetry
from nfn.common import NetworkSpec, WeightSpaceFeatures


class FlattenWeights(nn.Module):
    def __init__(self, network_spec):
        super().__init__()
        self.network_spec = network_spec

    def forward(self, wsfeat):
        wsfeat = shape_wsfeat_symmetry(wsfeat, self.network_spec)
        outs = []
        for w, b in wsfeat:
            outs.append(torch.flatten(w, start_dim=2).transpose(1, 2))
            outs.append(b.transpose(1, 2))
        return torch.cat(outs, dim=1)  # (B, N, C)


class UnflattenWeights(nn.Module):
    def __init__(self, network_spec: NetworkSpec):
        super().__init__()
        self.network_spec = network_spec

    def forward(self, x: torch.Tensor) -> WeightSpaceFeatures:
        # x.shape == (bs, num weights and biases)
        out_weights, out_biases = [], []
        curr_idx = 0
        for weight_spec, bias_spec in zip(self.network_spec.weight_spec, self.network_spec.bias_spec):
            num_wts = np.prod(weight_spec.shape)
            # reshape to (bs, 1, *weight_spec.shape) where 1 is channels.
            wt = x[:, curr_idx:curr_idx + num_wts].view(-1, *weight_spec.shape).unsqueeze(1)
            out_weights.append(wt)
            curr_idx += num_wts
            num_bs = np.prod(bias_spec.shape)
            bs = x[:, curr_idx:curr_idx + num_bs].view(-1, *bias_spec.shape).unsqueeze(1)
            out_biases.append(bs)
            curr_idx += num_bs
        return WeightSpaceFeatures(out_weights, out_biases)


class LearnedScale(nn.Module):
    def __init__(self, network_spec: NetworkSpec, init_scale):
        super().__init__()
        self.weight_scales = nn.ParameterList()
        self.bias_scales = nn.ParameterList()
        for _ in range(len(network_spec)):
            self.weight_scales.append(nn.Parameter(torch.tensor(init_scale, dtype=torch.float32)))
            self.bias_scales.append(nn.Parameter(torch.tensor(init_scale, dtype=torch.float32)))

    def forward(self, wsfeat: WeightSpaceFeatures) -> WeightSpaceFeatures:
        out_weights, out_biases = [], []
        for i, (weight, bias) in enumerate(zip(wsfeat.weights, wsfeat.biases)):
            out_weights.append(weight * self.weight_scales[i])
            out_biases.append(bias * self.bias_scales[i])
        return WeightSpaceFeatures(out_weights, out_biases)


class ResBlock(nn.Module):
    def __init__(self, base_layer, activation, dropout, norm):
        super().__init__()
        self.base_layer = base_layer
        self.activation = activation
        self.dropout = None
        if dropout > 0:
            self.dropout = TupleOp(nn.Dropout(dropout))
        self.norm = norm

    def forward(self, x: WeightSpaceFeatures) -> WeightSpaceFeatures:
        res = self.activation(self.base_layer(self.norm(x)))
        if self.dropout is not None:
            res = self.dropout(res)
        return x + res


class StatFeaturizer(nn.Module):
    def forward(self, wsfeat: WeightSpaceFeatures) -> torch.Tensor:
        out = []
        for (weight, bias) in wsfeat:
            out.append(self.compute_stats(weight))
            out.append(self.compute_stats(bias))
        return torch.cat(out, dim=-1)

    def compute_stats(self, tensor: torch.Tensor) -> torch.Tensor:
        """Computes the statistics of the given tensor."""
        tensor = torch.flatten(tensor, start_dim=2) # (B, C, H*W)
        mean = tensor.mean(-1) # (B, C)
        var = tensor.var(-1) # (B, C)
        q = torch.tensor([0., 0.25, 0.5, 0.75, 1.]).to(tensor.device)
        quantiles = torch.quantile(tensor, q, dim=-1) # (5, B, C)
        return torch.stack([mean, var, *quantiles], dim=-1) # (B, C, 7)

    @staticmethod
    def get_num_outs(network_spec):
        """Returns the number of outputs of the StatFeaturizer layer."""
        return 2 * len(network_spec) * 7


class TupleOp(nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op

    def forward(self, wsfeat: WeightSpaceFeatures) -> WeightSpaceFeatures:
        out_weights = [self.op(w) for w in wsfeat.weights]
        out_bias = [self.op(b) for b in wsfeat.biases]
        return WeightSpaceFeatures(out_weights, out_bias)