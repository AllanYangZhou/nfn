import numpy as np
import torch
from torch import nn
from functorch import make_functional, vmap


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
    func_model, _ = make_functional(siren)
    mgrid_len = 28 if dset_type == "mnist" else 32
    coords = get_mgrid(mgrid_len, 2).cuda()
    img_shape = IMG_shapes[dset_type]
    def func_inp(p):
        values = func_model(p, coords)[0]
        return torch.permute(values.reshape(*img_shape), (2, 0, 1))
    return vmap(func_inp, (0,))


def unprocess_img_arr(arr):
    arr = np.transpose(arr, (0, 2, 3, 1))
    if arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
    arr = np.clip(arr, -1, 1)
    return (255 * (arr + 1) / 2).astype(np.uint8)