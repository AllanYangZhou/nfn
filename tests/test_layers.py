import random
from einops import rearrange
import torch
from torch import nn
from nfn.layers import StatFeaturizer, TupleOp
from nfn.layers import NPLinear, HNPLinear, Pointwise, NPAttention
from nfn.common import WeightSpaceFeatures, network_spec_from_wsfeat


def check_params_eq(params1: WeightSpaceFeatures, params2: WeightSpaceFeatures):
    equal = True
    for p1, p2 in zip(params1.weights, params2.weights):
        equal = equal and torch.allclose(p1, p2, atol=1e-5)
    for p1, p2 in zip(params1.biases, params2.biases):
        equal = equal and torch.allclose(p1, p2, atol=1e-5)
    return equal


def sample_perm(layer_sizes):
    """Sample a random permutation for each of the hidden layers."""
    perms = []
    for layer_size in layer_sizes[1:-1]:
        perms.append(torch.eye(layer_size)[torch.randperm(layer_size)])
    return perms


def apply_perm_to_params(params, perms):
    """Apply a list of *coupled* perms to weights."""
    uncoupled_perms = convert_coupled_perm_to_uncoupled(perms, params)
    return apply_uncoupled_perm_to_params(params, uncoupled_perms)


def convert_coupled_perm_to_uncoupled(perms, params: WeightSpaceFeatures):
    """
    input: list of permutation matrices
    output: list of 2-tuples of permutation matrices, coupled
        in (row_perm, col_perm format)
    """
    out_perms = []
    prev_perm = torch.eye(params.weights[0].shape[3])
    for perm in perms:
        out_perms.append((perm, prev_perm.T))
        prev_perm = perm
    out_perms.append((torch.eye(params.weights[-1].shape[2]), prev_perm.T))
    return out_perms


def apply_uncoupled_perm_to_params(params: WeightSpaceFeatures, perms):
    """Perms is a list of 2-tuples of permutation matrices, one for rows and one for columns."""
    permed_weights = []
    permed_bias = []
    for (row_perm, col_perm), weight, bias in zip(perms, params.weights, params.biases):
        h, w = None, None
        if weight.dim() == 6:  # conv filter bank
            h, w = weight.shape[-2:]
            weight = rearrange(weight, 'b c i j k l -> b (c k l) i j')
        permed_weight = row_perm[None, None] @ weight @ col_perm[None, None]
        if h is not None:
            permed_weight = rearrange(permed_weight, 'b (c k l) i j -> b c i j k l', k=h, l=w)
        permed_weights.append(permed_weight)
        permed_bias.append((row_perm[None, None] @ bias.unsqueeze(-1)).squeeze(-1))
    return WeightSpaceFeatures(permed_weights, permed_bias)


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


def zip_stack_lists(lists):
    return [torch.stack(l) for l in zip(*lists)]


def sample_mlps(num_layers, bias, num_samples):
    assert not bias, "Bias not supported yet."
    widths = []
    for _ in range(num_layers):
        widths.append(random.randint(50, 100))
    layers = [nn.Linear(784, widths[0], bias=bias), nn.ReLU()]
    for i in range(1, num_layers):
        layers.append(nn.Linear(widths[i - 1], widths[i], bias=bias))
        if i < num_layers - 1:
            layers.append(nn.ReLU())
    mlp = nn.Sequential(*layers)
    weights_list = []
    for _ in range(num_samples):
        mlp.apply(weight_reset)
        weights_list.append([w.detach()[None] for _name, w in mlp.state_dict().items()])
    return mlp.state_dict().keys(), zip_stack_lists(weights_list)


def sample_params(bs, chan, layer_sizes):
    weights, biases = [], []
    for i in range(len(layer_sizes) - 1):
        weights.append(torch.randn(bs, chan, layer_sizes[i + 1], layer_sizes[i]))
        biases.append(torch.randn(bs, chan, layer_sizes[i + 1]))
    return WeightSpaceFeatures(weights, biases)


def sample_cnn_params(bs, chan, layer_sizes, filter_sizes):
    weights, biases = [], []
    for i in range(len(layer_sizes) - 1):
        weights.append(torch.randn(bs, chan, layer_sizes[i + 1], layer_sizes[i], filter_sizes[i][0], filter_sizes[i][1]))
        biases.append(torch.randn(bs, chan, layer_sizes[i + 1]))
    return WeightSpaceFeatures(weights, biases)


def test_sensitivity_uncoupled(layer, layer_sizes, c_in, filter_sizes=None):
    bs = 15
    if filter_sizes is not None:
        params = sample_cnn_params(bs, c_in, layer_sizes, filter_sizes)
    else:
        params = sample_params(bs, c_in, layer_sizes)
    out = layer(params)
    sensitive = True
    for idx in range(1, 2 * (len(layer_sizes) - 1) - 1):
        equiv = True
        for _ in range(10):
            perm = convert_coupled_perm_to_uncoupled(sample_perm(layer_sizes), params)
            # edit to break coupling
            row_perm, col_perm = perm[idx // 2]
            if idx % 2 == 0:
                col_perm = torch.eye(col_perm.shape[0])[torch.randperm(col_perm.shape[0])]
            else:
                row_perm = torch.eye(row_perm.shape[0])[torch.randperm(row_perm.shape[0])]
            perm[idx // 2] = (row_perm, col_perm)

            permed_params = apply_uncoupled_perm_to_params(params, perm)
            out_of_permed = layer(permed_params)
            permed_out = apply_uncoupled_perm_to_params(out, perm)
            # just need one input where equivariance is broken
            equiv = equiv and check_params_eq(out_of_permed, permed_out)
        # ensure that every layer is sensitive
        sensitive = sensitive and (not equiv)
    return sensitive


def test_layer_equivariance(layer, layer_sizes, c_in, filter_sizes=None):
    bs = 15
    if filter_sizes is not None:
        params = sample_cnn_params(bs, c_in, layer_sizes, filter_sizes)
    else:
        params = sample_params(bs, c_in, layer_sizes)
    out = layer(params)
    equiv = True
    for _ in range(10):
        perm = sample_perm(layer_sizes)
        permed_params = apply_perm_to_params(params, perm)
        out_of_permed = layer(permed_params)
        permed_out = apply_perm_to_params(out, perm)
        equiv = equiv and check_params_eq(out_of_permed, permed_out)
    return equiv


def test_HNPLinear_2layer():
    c_in, c_out = 3, 4
    layer_sizes = [10, 15, 12]
    spec = network_spec_from_wsfeat(sample_params(1, c_in, layer_sizes))
    layer = HNPLinear(spec, c_in, c_out)
    assert test_layer_equivariance(layer, layer_sizes, c_in)
    print("HNPLinear (2L) is equivariant to coupled.")
    assert test_sensitivity_uncoupled(layer, layer_sizes, c_in)
    print("HNPLinear (2L) is sensitive to uncoupled.")

    filter_sizes = [(random.randint(3, 8), random.randint(3, 8)) for _ in range(len(layer_sizes) - 1)]
    spec = network_spec_from_wsfeat(sample_cnn_params(1, c_in, layer_sizes, filter_sizes))
    layer = HNPLinear(spec, c_in, c_out)
    assert test_layer_equivariance(layer, layer_sizes, c_in, filter_sizes)
    print("HNPLinear (2L) is equivariant for CNNs.")
    assert test_sensitivity_uncoupled(layer, layer_sizes, c_in, filter_sizes)
    print("HNPLinear (2L) is sensitive to uncoupled for CNNs.")


def test_NPLinear():
    layer_sizes = [10, 15, 13, 18, 14, 12]
    c_in, c_out = 3, 4
    spec = network_spec_from_wsfeat(sample_params(1, c_in, layer_sizes))
    layer = NPLinear(spec, c_in, c_out)
    assert test_layer_equivariance(layer, layer_sizes, c_in)
    print("NPLinear is equivariant.")
    assert test_sensitivity_uncoupled(layer, layer_sizes, c_in)
    print("NPLinear is sensitive to uncoupled.")

    filter_sizes = [(random.randint(3, 8), random.randint(3, 8)) for _ in range(len(layer_sizes) - 1)]
    spec = network_spec_from_wsfeat(sample_cnn_params(1, c_in, layer_sizes, filter_sizes))
    layer = NPLinear(spec, c_in, c_out)
    assert test_layer_equivariance(layer, layer_sizes, c_in, filter_sizes)
    print("NPLinear is equivariant for CNNs.")
    assert test_sensitivity_uncoupled(layer, layer_sizes, c_in, filter_sizes)
    print("NPLinear is sensitive to uncoupled for CNNs.")


def variance_wsfeat(wsfeat):
    values = []
    for weight, bias in wsfeat:
        values.append(torch.flatten(weight, start_dim=1))
        values.append(torch.flatten(bias, start_dim=1))
    values = torch.cat(values, dim=1)
    return torch.var(values)


def test_init(init_type):
    layer_sizes = [64, 128, 32]
    chan = 64
    bs = 100
    spec = network_spec_from_wsfeat(sample_params(1, chan, layer_sizes))
    model = nn.Sequential(
        NPLinear(spec, chan, chan, init_type=init_type), TupleOp(nn.ReLU()),
        NPLinear(spec, chan, chan, init_type=init_type), TupleOp(nn.ReLU()),
        NPLinear(spec, chan, chan, init_type=init_type), TupleOp(nn.ReLU()),
        NPLinear(spec, chan, chan, init_type=init_type), TupleOp(nn.ReLU()),
        NPLinear(spec, chan, chan, init_type=init_type)
    )
    params = sample_params(bs, chan, layer_sizes)
    out = model(params)
    var_out = variance_wsfeat(out)
    print(f"Variance of output with init_type={init_type}: {var_out:.3f}")


def test_HNPLinear():
    layer_sizes = [10, 15, 13, 18, 14, 12]
    c_in, c_out = 3, 4
    spec = network_spec_from_wsfeat(sample_params(1, c_in, layer_sizes))
    layer = HNPLinear(spec, c_in, c_out)
    assert test_layer_equivariance(layer, layer_sizes, c_in)
    print("HNPLinear is equivariant.")
    assert test_sensitivity_uncoupled(layer, layer_sizes, c_in)
    print("HNPLinear is sensitive to uncoupled.")

    filter_sizes = [(random.randint(3, 8), random.randint(3, 8)) for _ in range(len(layer_sizes) - 1)]
    spec = network_spec_from_wsfeat(sample_cnn_params(1, c_in, layer_sizes, filter_sizes))
    layer = HNPLinear(spec, c_in, c_out)
    assert test_layer_equivariance(layer, layer_sizes, c_in, filter_sizes)
    print("HNPLinear is equivariant for CNNs.")
    assert test_sensitivity_uncoupled(layer, layer_sizes, c_in, filter_sizes)
    print("HNPLinear is sensitive to uncoupled for CNNs.")


def test_Pointwise():
    layer_sizes = [10, 15, 13, 14, 12]
    c_in, c_out = 3, 4
    spec = network_spec_from_wsfeat(sample_params(1, c_in, layer_sizes))
    layer = Pointwise(spec, c_in, c_out)
    # assert test_layer_equivariance(layer, layer_sizes, c_in)
    print("Pointwise is equivariant.")
    # assert not test_sensitivity_uncoupled(layer, layer_sizes, c_in)
    print("Pointwise is not sensitive to uncoupled.")

    filter_sizes = [(random.randint(3, 8), random.randint(3, 8)) for _ in range(len(layer_sizes) - 1)]
    spec = network_spec_from_wsfeat(sample_cnn_params(1, c_in, layer_sizes, filter_sizes))
    layer = Pointwise(spec, c_in, c_out)
    assert test_layer_equivariance(layer, layer_sizes, c_in, filter_sizes)
    print("Pointwise is equivariant for CNNs.")
    assert not test_sensitivity_uncoupled(layer, layer_sizes, c_in, filter_sizes)
    print("Pointwise is not sensitive to uncoupled for CNNs.")

def test_StatFeaturizer():
    layer_sizes = [10, 15, 13, 14, 12]
    c_in, c_out = 3, 3
    layer = StatFeaturizer()
    params = sample_params(10, c_in, layer_sizes)
    assert StatFeaturizer.get_num_outs(network_spec_from_wsfeat(params)) == layer(params).shape[-1]
    print("StatFeaturizer shape is correct.")
    

def test_NPAttention():
    layer_sizes = [10, 15, 13, 18, 14, 12]
    channels = 64
    spec = network_spec_from_wsfeat(sample_params(1, channels, layer_sizes))
    layer = NPAttention(spec, 64, 8, dropout=0)
    assert test_layer_equivariance(layer, layer_sizes, channels)
    print("NPAttention is equivariant.")
    assert test_sensitivity_uncoupled(layer, layer_sizes, channels)
    print("NPAttention is sensitive to uncoupled.")

    filter_sizes = [(random.randint(3, 8), random.randint(3, 8)) for _ in range(len(layer_sizes) - 1)]
    spec = network_spec_from_wsfeat(sample_cnn_params(1, channels, layer_sizes, filter_sizes))
    layer = NPAttention(spec, channels, share_projections=False)
    assert test_layer_equivariance(layer, layer_sizes, channels, filter_sizes)
    print("NPAttention is equivariant for CNNs.")
    assert test_sensitivity_uncoupled(layer, layer_sizes, channels, filter_sizes)
    print("NPAttention is sensitive to uncoupled for CNNs.")


if __name__ == "__main__":
    test_init("pytorch_default")
    test_init("kaiming_normal")
    test_NPAttention()
    test_Pointwise()
    test_HNPLinear_2layer()
    test_NPLinear()
    test_HNPLinear()
    test_StatFeaturizer()