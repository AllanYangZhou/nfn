import math
import numpy as np
import torch
from torch import nn
from functorch import make_functional, vmap
from nfn.common import NetworkSpec


def get_mgrid(sidelen, dim=2):
    """Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int"""
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(
        self, in_features, out_features, bias=True, is_first=False, omega_0=30
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        outermost_linear=False,
        first_omega_0=30,
        hidden_omega_0=30.0,
    ):
        super().__init__()

        self.net = []
        self.net.append(
            SineLayer(
                in_features, hidden_features, is_first=True, omega_0=first_omega_0
            )
        )

        for i in range(hidden_layers):
            self.net.append(
                SineLayer(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(
                    -np.sqrt(6 / hidden_features) / hidden_omega_0,
                    np.sqrt(6 / hidden_features) / hidden_omega_0,
                )

            self.net.append(final_linear)
        else:
            self.net.append(
                SineLayer(
                    hidden_features,
                    out_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        output = self.net(coords)
        return output, coords


IMG_shapes = {"mnist": (28, 28, 1), "cifar": (32, 32, 3)}
SIREN_kwargs = {
    "mnist": {
        "first_omega_0": 30,
        "hidden_features": 32,
        "hidden_layers": 1,
        "hidden_omega_0": 30.0,
        "in_features": 2,
        "out_features": 1,
        "outermost_linear": True,
    },
    "cifar": {
        "first_omega_0": 30,
        "hidden_features": 32,
        "hidden_layers": 1,
        "hidden_omega_0": 30.0,
        "in_features": 2,
        "out_features": 3,
        "outermost_linear": True,
    },
}
def get_batch_siren(dset_type):
    siren = Siren(**SIREN_kwargs[dset_type])
    func_model, initial_params = make_functional(siren)
    mgrid_len = 28 if dset_type in ["mnist", "fashion"] else 32
    coords = get_mgrid(mgrid_len, 2).cuda()
    img_shape = IMG_shapes[dset_type]
    def func_inp(p):
        values = func_model(p, coords)[0]
        return torch.permute(values.reshape(*img_shape), (2, 0, 1))
    return vmap(func_inp, (0,)), initial_params


def get_spatial_batch_siren(dset_type):
    siren = Siren(**SIREN_kwargs[dset_type])
    func_model, initial_params = make_functional(siren)
    mgrid_len = 28 if dset_type in ["mnist", "fashion"] else 32
    coords = get_mgrid(mgrid_len, 2).cuda()
    img_shape = IMG_shapes[dset_type]
    def siren_at_coord(params, coord):
        # TODO: this should be batched over all coords that correspond to this param patch.
        # evaluate siren at a single coordinate
        return func_model(params, coord[None])[0].squeeze(0)
    batch_siren_at_coord = vmap(siren_at_coord, (0, 0))
    def spatial_siren(params):
        # params is a list of tensors, each of shape (width^2, ...).
        width = int(math.sqrt(params[0].shape[0]))
        def reshape(arr):
            return arr.reshape(width, width, *arr.shape[1:])
        # each param now has shape (width, width, ...)
        params = [reshape(p) for p in params]
        param_idcs = torch.round((params[0].shape[0] - 1) * (coords + 1) / 2).long().cpu()
        idxd_params = [p[param_idcs[:, 0], param_idcs[:, 1]] for p in params]
        values = batch_siren_at_coord(idxd_params, coords)
        return torch.permute(values.reshape(*img_shape), (2, 0, 1))
    return vmap(spatial_siren, (0,)), initial_params


def unprocess_img_arr(arr):
    arr = np.transpose(arr, (0, 2, 3, 1))
    if arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
    arr = np.clip(arr, -1, 1)
    return (255 * (arr + 1) / 2).astype(np.uint8)


def hyper_weight_init(m, in_features_main_net):
    if hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        m.weight.data = m.weight.data / 1.e2

    if hasattr(m, 'bias'):
        with torch.no_grad():
            m.bias.uniform_(-1/in_features_main_net, 1/in_features_main_net)


def hyper_bias_init(m):
    if hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        m.weight.data = m.weight.data / 1.e2

    if hasattr(m, 'bias'):
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        with torch.no_grad():
            m.bias.uniform_(-1/fan_in, 1/fan_in)


class SimpleHyperNetwork(nn.Module):
    def __init__(self, network_spec: NetworkSpec, in_size, hidden_size=256, hidden_layers=1):
        super().__init__()
        self.network_spec = network_spec
        layers = [nn.Linear(in_size, hidden_size), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        num_params = network_spec.get_num_params()
        layers.append(nn.Linear(hidden_size, num_params))
        self.hnet = nn.Sequential(*layers)

    def forward(self, x):
        return self.hnet(x)


class HyperNetwork(nn.Module):
    def __init__(self, network_spec: NetworkSpec, in_size, hidden_size=256, hidden_layers=1):
        super().__init__()
        self.network_spec = network_spec
        self.hnets = nn.ModuleList([])
        # construct a hypernetwork for each weight and bias in each layer
        for i in range(len(network_spec)):
            self.hnets.append(self._get_hnet(in_size, network_spec.weight_spec[i].shape, hidden_size, hidden_layers))
            self.hnets.append(self._get_hnet(in_size, network_spec.bias_spec[i].shape, hidden_size, hidden_layers))

    def _get_hnet(self, in_size, out_shape, hidden_size, hidden_layers=1):
        out_size = np.prod(out_shape)
        hnet = [nn.Linear(in_size, hidden_size), nn.ReLU()]
        for _ in range(hidden_layers):
            hnet.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        hnet.append(nn.Linear(hidden_size, out_size))
        hnet = nn.Sequential(*hnet)
        # Following: https://github.com/vsitzmann/siren/blob/master/meta_modules.py#L135
        if len(out_shape) == 1:  # outputting bias
            hnet[-1].apply(hyper_bias_init)
        else:  # outputting weight
            hnet[-1].apply(lambda m: hyper_weight_init(m, out_shape[-1]))
        return hnet

    def forward(self, x):
        outs = []
        for hnet in self.hnets:
            outs.append(hnet(x))
        return torch.cat(outs, dim=1)