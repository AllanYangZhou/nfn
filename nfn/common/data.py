from dataclasses import dataclass
from typing import List, Tuple
import collections
from collections import OrderedDict


@dataclass(frozen=True)
class ArraySpec:
    shape: Tuple[int, ...]


@dataclass(frozen=True)
class NetworkSpec:
    weight_spec: List[ArraySpec]
    bias_spec: List[ArraySpec]

    def get_io(self):
        # n_in, n_out
        return self.weight_spec[0].shape[1], self.weight_spec[-1].shape[0]
    
    def get_num_params(self):
        """Returns the number of parameters in the network."""
        num_params = 0
        for w, b in zip(self.weight_spec, self.bias_spec):
            num_weights = 1
            for dim in w.shape:
                assert dim != -1
                num_weights *= dim
            num_biases = 1
            for dim in b.shape:
                assert dim != -1
                num_biases *= dim
            num_params += num_weights+num_biases
        return num_params

    def __len__(self):
        return len(self.weight_spec)


class WeightSpaceFeatures(collections.abc.Sequence):
    def __init__(self, weights, biases):
        # No mutability
        if isinstance(weights, list): weights = tuple(weights)
        if isinstance(biases, list): biases = tuple(biases)
        self.weights = weights
        self.biases = biases

    def __len__(self):
        return len(self.weights)

    def __iter__(self):
        return zip(self.weights, self.biases)

    def __getitem__(self, idx):
        return (self.weights[idx], self.biases[idx])

    def __add__(self, other):
        out_weights = tuple(w1 + w2 for w1, w2 in zip(self.weights, other.weights))
        out_biases = tuple(b1 + b2 for b1, b2 in zip(self.biases, other.biases))
        return WeightSpaceFeatures(out_weights, out_biases)

    def __mul__(self, other):
        if isinstance(other, WeightSpaceFeatures):
            weights = tuple(w1 * w2 for w1, w2 in zip(self.weights, other.weights))
            biases = tuple(b1 * b2 for b1, b2 in zip(self.biases, other.biases))
            return WeightSpaceFeatures(weights, biases)
        return self.map(lambda x: x * other)

    def detach(self):
        """Returns a copy with detached tensors."""
        return WeightSpaceFeatures(tuple(w.detach() for w in self.weights), tuple(b.detach() for b in self.biases))

    def map(self, func):
        """Applies func to each weight and bias tensor."""
        return WeightSpaceFeatures(tuple(func(w) for w in self.weights), tuple(func(b) for b in self.biases))

    def to(self, device):
        """Moves all tensors to device."""
        return WeightSpaceFeatures(tuple(w.to(device) for w in self.weights), tuple(b.to(device) for b in self.biases))

    @classmethod
    def from_zipped(cls, weight_and_biases):
        """Converts a list of (weights, biases) into a WeightSpaceFeatures object."""
        weights, biases = zip(*weight_and_biases)
        return cls(weights, biases)


def state_dict_to_tensors(state_dict):
    """Converts a state dict into two lists of equal length:
    1. list of weight tensors
    2. list of biases, or None if no bias
    Assumes the state_dict key order is [0.weight, 0.bias, 1.weight, 1.bias, ...]
    """
    weights, biases = [], []
    keys = list(state_dict.keys())
    i = 0
    while i < len(keys):
        weights.append(state_dict[keys[i]][None])
        i += 1
        assert keys[i].endswith("bias")
        biases.append(state_dict[keys[i]][None])
        i += 1
    return weights, biases


def params_to_state_dicts(keys, wsfeat: WeightSpaceFeatures) -> List[OrderedDict]:
    """Converts a list of weight tensors and a list of biases into a state dict.
    Assumes the state_dict key order is [0.weight, 0.bias, 1.weight, 1.bias, ...]
    """
    batch_size = wsfeat.weights[0].shape[0]
    assert wsfeat.weights[0].shape[1] == 1
    state_dicts = [OrderedDict() for _ in range(batch_size)]
    layer_idx = 0
    while layer_idx < len(keys):
        for batch_idx in range(batch_size):
            state_dicts[batch_idx][keys[layer_idx]] = wsfeat.weights[layer_idx // 2][batch_idx].squeeze(0)
        layer_idx += 1
        for batch_idx in range(batch_size):
            state_dicts[batch_idx][keys[layer_idx]] = wsfeat.biases[layer_idx // 2][batch_idx].squeeze(0)
        layer_idx += 1
    return state_dicts


def network_spec_from_wsfeat(wsfeat: WeightSpaceFeatures, set_all_dims=False) -> NetworkSpec:
    assert len(wsfeat.weights) == len(wsfeat.biases)
    weight_specs = []
    bias_specs = []
    for i, (weight, bias) in enumerate(zip(wsfeat.weights, wsfeat.biases)):
        # -1 means the dimension has symmetry, hence summed/broadcast over.
        # Recall that weight has two leading dimension (BS, channels, ...)
        if weight.dim() == 4:
            weight_shape = [-1, -1]
        elif weight.dim() == 6:
            weight_shape = [-1, -1, weight.shape[-2], weight.shape[-1]]
        else:
            raise ValueError(f"Unsupported weight dim: {weight.dim()}")
        if i == 0 or set_all_dims: weight_shape[1] = weight.shape[3]
        if i == len(wsfeat) - 1 or set_all_dims: weight_shape[0] = weight.shape[2]
        weight_specs.append(ArraySpec(tuple(weight_shape)))
        bias_shape = (-1,)
        if i == len(wsfeat) - 1 or set_all_dims:
            bias_shape = (bias.shape[-1],)
        bias_specs.append(ArraySpec(bias_shape))
    return NetworkSpec(weight_specs, bias_specs)
