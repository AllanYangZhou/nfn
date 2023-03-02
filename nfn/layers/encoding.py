import math
import torch
from torch import nn
from nfn.common import WeightSpaceFeatures, NetworkSpec


class GaussianFourierFeatureTransform(nn.Module):
    """
    Given an input of size [batches, num_input_channels, ...],
     returns a tensor of size [batches, mapping_size*2, ...].
    """

    def __init__(self, network_spec, in_channels, mapping_size=256, scale=10):
        super().__init__()
        del network_spec
        self._num_input_channels = in_channels
        self._mapping_size = mapping_size
        self.out_channels = mapping_size * 2
        self.register_buffer("_B", torch.randn((in_channels, mapping_size)) * scale)

    def encode_tensor(self, x):
        assert len(x.shape) >= 3
        # Put channels dimension last.
        x = (x.transpose(1, -1) @ self._B).transpose(1, -1)
        x = 2 * math.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)

    def forward(self, wsfeat):
        out_weights, out_biases = [], []
        for (weight, bias) in wsfeat:
            out_weights.append(self.encode_tensor(weight))
            out_biases.append(self.encode_tensor(bias))
        return WeightSpaceFeatures(out_weights, out_biases)


def fourier_encode(x, max_freq, num_bands = 4):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.linspace(1., max_freq / 2, num_bands, device = device, dtype = dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * math.pi
    x = torch.cat([x.sin(), x.cos()], dim = -1)
    x = torch.cat((x, orig_x), dim = -1)
    return x


class IOSinusoidalEncoding(nn.Module):
    def __init__(self, network_spec: NetworkSpec, max_freq=10, num_bands=6, enc_layers=True):
        super().__init__()
        self.network_spec = network_spec
        self.max_freq = max_freq
        self.num_bands = num_bands
        self.enc_layers = enc_layers
        self.n_in, self.n_out = network_spec.get_io()

    def forward(self, wsfeat: WeightSpaceFeatures):
        device, dtype = wsfeat.weights[0].device, wsfeat.weights[0].dtype
        L = len(self.network_spec)
        layernum = torch.linspace(-1., 1., steps=L, device=device, dtype=dtype)
        if self.enc_layers:
            layer_enc = fourier_encode(layernum, self.max_freq, self.num_bands)  # (L, 2 * num_bands + 1)
        else:
            layer_enc = torch.zeros((L, 2 * self.num_bands + 1), device=device, dtype=dtype)
        inpnum = torch.linspace(-1., 1., steps=self.n_in, device=device, dtype=dtype)
        inp_enc = fourier_encode(inpnum, self.max_freq, self.num_bands)  # (n_in, 2 * num_bands + 1)
        outnum = torch.linspace(-1., 1., steps=self.n_out, device=device, dtype=dtype)
        out_enc = fourier_encode(outnum, self.max_freq, self.num_bands)  # (n_out, 2 * num_bands + 1)

        d = 2 * self.num_bands + 1

        out_weights, out_biases = [], []
        for i, (weight, bias) in enumerate(wsfeat):
            b, _, *axes = weight.shape
            enc_i = layer_enc[i].unsqueeze(0)[..., None, None]
            for _ in axes[2:]:
                enc_i = enc_i.unsqueeze(-1)
            enc_i = enc_i.expand(b, d, *axes) # (B, d, n_row, n_col, ...)
            bias_enc_i = layer_enc[i][None, :, None].expand(b, d, bias.shape[-1])  # (B, d, n_row)
            if i == 0:
                # weight has shape (B, c_in, n_out, n_in)
                inp_enc_i = inp_enc.transpose(0, 1).unsqueeze(0).unsqueeze(-2)  # (1, d, 1, n_col)
                for _ in axes[2:]:
                    inp_enc_i = inp_enc_i.unsqueeze(-1)
                enc_i = enc_i  + inp_enc_i
            if i == len(wsfeat) - 1:
                out_enc_i = out_enc.transpose(0, 1).unsqueeze(0).unsqueeze(-1)  # (1, d, n_row, 1)
                for _ in axes[2:]:
                    out_enc_i = inp_enc_i.unsqueeze(-1)
                enc_i = enc_i  + out_enc_i
                bias_enc_i = bias_enc_i + out_enc.transpose(0, 1).unsqueeze(0)
            out_weights.append(torch.cat([weight, enc_i], dim=1))
            out_biases.append(torch.cat([bias, bias_enc_i], dim=1))
        return WeightSpaceFeatures(out_weights, out_biases)

    def num_out_chan(self, in_chan):
        return in_chan + (2 * self.num_bands + 1)