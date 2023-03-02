import numpy as np
import torch
from torch import nn
from einops import rearrange
from nfn.common import NetworkSpec, WeightSpaceFeatures
from nfn.layers.layer_utils import set_init_, shape_wsfeat_symmetry, unshape_wsfeat_symmetry


class NPPool(nn.Module):
    def __init__(self, network_spec: NetworkSpec, agg="mean"):
        super().__init__()
        self.network_spec = network_spec
        if agg == "mean":
            self.agg = lambda x, dim: x.mean(dim=dim)
        elif agg == "max":
            self.agg = lambda x, dim: x.amax(dim=dim)
        elif agg == "sum":
            self.agg = lambda x, dim: x.sum(dim=dim)
        else:
            raise NotImplementedError(f"Aggregation {agg} not implemented.")

    def forward(self, wsfeat: WeightSpaceFeatures) -> torch.Tensor:
        out = []
        for weight, bias in wsfeat:
            out.append(self.agg(weight, dim=(2,3)).unsqueeze(-1))
            out.append(self.agg(bias, dim=-1).unsqueeze(-1))
        return torch.cat([torch.flatten(o, start_dim=2) for o in out], dim=-1)

    @staticmethod
    def get_num_outs(network_spec):
        """Returns the number of outputs of the global pooling layer."""
        filter_facs = [int(np.prod(spec.shape[2:])) for spec in network_spec.weight_spec]
        num_outs = 0
        for fac in filter_facs:
            num_outs += fac + 1
        return num_outs


class HNPPool(nn.Module):
    def __init__(self, network_spec: NetworkSpec, agg="mean"):
        super().__init__()
        self.network_spec = network_spec
        if agg == "mean":
            self.agg = lambda x, dim: x.mean(dim=dim)
        elif agg == "max":
            self.agg = lambda x, dim: x.amax(dim=dim)
        elif agg == "sum":
            self.agg = lambda x, dim: x.sum(dim=dim)
        else:
            raise NotImplementedError(f"Aggregation {agg} not implemented.")

    def forward(self, wsfeat: WeightSpaceFeatures) -> torch.Tensor:
        out = []
        for i, (weight, bias) in enumerate(wsfeat):
            if i == 0:
                out.append(self.agg(weight, dim=2))  # average over rows
            elif i == len(wsfeat) - 1:
                out.append(self.agg(weight, dim=3))  # average over cols
            else:
                out.append(self.agg(weight, dim=(2,3)).unsqueeze(-1))
            if i == len(wsfeat) - 1: out.append(bias)
            else: out.append(self.agg(bias, dim=-1).unsqueeze(-1))
        return torch.cat([torch.flatten(o, start_dim=2) for o in out], dim=-1)

    @staticmethod
    def get_num_outs(network_spec):
        """Returns the number of outputs of the global pooling layer."""
        filter_facs = [int(np.prod(spec.shape[2:])) for spec in network_spec.weight_spec]
        n_in, n_out = network_spec.get_io()
        num_outs = 0
        for i, fac in enumerate(filter_facs):
            if i == 0:
                num_outs += n_in * fac + 1
            elif i == len(filter_facs) - 1:
                num_outs += n_out * fac + n_out
            else:
                num_outs += fac + 1
        return num_outs


class Pointwise(nn.Module):
    """Assumes full row/col exchangeability of weights in each layer."""
    def __init__(self, network_spec: NetworkSpec, in_channels, out_channels):
        super().__init__()
        self.network_spec = network_spec
        self.in_channels = in_channels
        self.out_channels = out_channels
        # register num_layers in_channels -> out_channels linear layers
        filter_facs = [int(np.prod(spec.shape[2:])) for spec in network_spec.weight_spec]
        for i in range(len(network_spec)):
            fac_i = filter_facs[i]
            self.add_module(f"layer_{i}", nn.Conv2d(fac_i * in_channels, fac_i * out_channels, 1))
            self.add_module(f"bias_{i}", nn.Conv1d(in_channels, out_channels, 1))

    def forward(self, wsfeat: WeightSpaceFeatures):
        wsfeat = shape_wsfeat_symmetry(wsfeat, self.network_spec)
        # weights is a list of tensors, each with shape (B, C_in, nrow, ncol)
        # each tensor is reshaped to (B, nrow * ncol, C_in), passed through a linear
        # layer, then reshaped back to (B, C_out, nrow, ncol)
        out_weights, out_biases = [], []
        for i, (weight, bias) in enumerate(wsfeat):
            out_weights.append(getattr(self, f"layer_{i}")(weight))
            out_biases.append(getattr(self, f"bias_{i}")(bias))
        return unshape_wsfeat_symmetry(WeightSpaceFeatures(out_weights, out_biases), self.network_spec)

    def __repr__(self):
        return f"Pointwise(in_channels={self.in_channels}, out_channels={self.out_channels})"


class NPLinear(nn.Module):
    """Assume permutation symmetry of input and output layers, as well as hidden."""
    def __init__(self, network_spec: NetworkSpec, in_channels, out_channels, io_embed=False):
        super().__init__()
        self.c_in, self.c_out = in_channels, out_channels
        self.network_spec = network_spec
        L = len(network_spec)
        filter_facs = [int(np.prod(spec.shape[2:])) for spec in network_spec.weight_spec]
        n_rc_inp = L + sum(filter_facs)
        for i in range(L):
            fac_i = filter_facs[i]
            # pointwise
            self.add_module(f"layer_{i}", nn.Conv2d(fac_i * in_channels, fac_i * out_channels, 1))
            # broadcasts over rows and columns
            self.add_module(f"layer_{i}_rc", nn.Linear(n_rc_inp * in_channels, fac_i * out_channels))

            # broadcasts over rows or columns
            row_in, col_in = fac_i * in_channels, (fac_i + 1) * in_channels
            if i > 0:
                fac_im1 = filter_facs[i - 1]
                row_in += (fac_im1 + 1) * in_channels
            if i < len(network_spec) - 1:
                fac_ip1 = filter_facs[i + 1]
                col_in += fac_ip1 * in_channels
            self.add_module(f"layer_{i}_r", nn.Conv1d(row_in, fac_i * out_channels, 1))
            self.add_module(f"layer_{i}_c", nn.Conv1d(col_in, fac_i * out_channels, 1))

            # pointwise
            self.add_module(f"bias_{i}", nn.Conv1d(col_in, out_channels, 1))
            self.add_module(f"bias_{i}_rc", nn.Linear(n_rc_inp * in_channels, out_channels))
            set_init_(
                getattr(self, f"layer_{i}"),
                getattr(self, f"layer_{i}_rc"),
                getattr(self, f"layer_{i}_r"),
                getattr(self, f"layer_{i}_c"),
            )
            set_init_(getattr(self, f"bias_{i}"), getattr(self, f"bias_{i}_rc"))
        self.io_embed = io_embed
        if io_embed:
            # initialize learned position embeddings to break input and output symmetry
            n_in, n_out = network_spec.get_io()
            self.in_embed = nn.Parameter(torch.randn(1, filter_facs[0] * in_channels, 1, n_in))
            self.out_embed = nn.Parameter(torch.randn(1, filter_facs[-1] * in_channels, n_out, 1))
            self.out_bias_embed = nn.Parameter(torch.randn(1, in_channels, n_out))

    def forward(self, wsfeat: WeightSpaceFeatures) -> WeightSpaceFeatures:
        wsfeat = shape_wsfeat_symmetry(wsfeat, self.network_spec)
        if self.io_embed:
            new_weights = (wsfeat.weights[0] + self.in_embed, *wsfeat.weights[1:-1], wsfeat.weights[-1] + self.out_embed)
            new_biases = (*wsfeat.biases[:-1], wsfeat.biases[-1] + self.out_bias_embed)
            wsfeat = WeightSpaceFeatures(new_weights, new_biases)
        weights, biases = wsfeat.weights, wsfeat.biases
        row_means = [w.mean(dim=-2) for w in weights]
        col_means = [w.mean(dim=-1) for w in weights]
        rowcol_means = [w.mean(dim=(-2, -1)) for w in weights]  # (B, C_in)
        bias_means = [b.mean(dim=-1) for b in biases]  # (B, C_in)
        wb_means = torch.cat(rowcol_means + bias_means, dim=-1)  # (B, 2 * C_in * n_layers)
        out_weights, out_biases = [], []
        for i, (weight, bias) in enumerate(wsfeat):
            z1 = getattr(self, f"layer_{i}")(weight)  # (B, C_out, nrow, ncol)
            z2 = getattr(self, f"layer_{i}_rc")(wb_means)[..., None, None]
            row_bdcst = [row_means[i]]  # (B, C_in, ncol)
            col_bdcst = [col_means[i], bias]  # (B, 2 * C_in, nrow)
            if i > 0:
                row_bdcst.extend([col_means[i-1], biases[i-1]])  # (B, C_in, ncol)
            if i < len(wsfeat.weights) - 1:
                col_bdcst.append(row_means[i+1])  # (B, C_in, nrow)
            col_bdcst = torch.cat(col_bdcst, dim=-2)
            z3 = getattr(self, f"layer_{i}_r")(torch.cat(row_bdcst, dim=-2)).unsqueeze(-2)  # (B, C_out, 1, ncol)
            z4 = getattr(self, f"layer_{i}_c")(col_bdcst).unsqueeze(-1)  # (B, C_out, nrow, 1)
            out_weights.append(z1 + z2 + z3 + z4)

            u1 = getattr(self, f"bias_{i}")(col_bdcst)  # (B, C_out, nrow)
            u2 = getattr(self, f"bias_{i}_rc")(wb_means).unsqueeze(-1)  # (B, C_out, 1)
            out_biases.append(u1 + u2)
        return unshape_wsfeat_symmetry(WeightSpaceFeatures(out_weights, out_biases), self.network_spec)

    def __repr__(self):
        return f"NPLinear(in_channels={self.c_in}, out_channels={self.c_out})"


class HNPLinear(nn.Module):
    def __init__(self, network_spec: NetworkSpec, in_channels, out_channels):
        super().__init__()
        self.network_spec = network_spec
        n_in, n_out = network_spec.get_io()
        self.n_in, self.n_out = n_in, n_out
        self.c_in, self.c_out = in_channels, out_channels
        self.L = len(network_spec)
        filter_facs = [int(np.prod(spec.shape[2:])) for spec in network_spec.weight_spec]
        n_rc_inp = n_in * filter_facs[0] + n_out * filter_facs[-1] + self.L + n_out
        for i in range(self.L):
            n_rc_inp += filter_facs[i]
        for i in range(self.L):
            fac_i = filter_facs[i]
            if i == 0:
                fac_ip1 = filter_facs[i+1]
                rpt_in = (fac_i * n_in + fac_ip1 + 1)
                if i + 1 == self.L - 1:
                    rpt_in += n_out * fac_ip1
                self.w0_rpt = nn.Conv1d(rpt_in * in_channels, n_in * fac_i * out_channels, 1)
                self.w0_rbdcst = nn.Linear(n_rc_inp * in_channels, n_in * fac_i * out_channels)

                self.bias_0 = nn.Conv1d(rpt_in * in_channels, out_channels, 1)
                self.bias_0_rc = nn.Linear(n_rc_inp * in_channels, out_channels)
                set_init_(self.bias_0, self.bias_0_rc)
            elif i == self.L - 1:
                fac_im1 = filter_facs[i-1]
                cpt_in = (fac_i * n_out + fac_im1)
                if i - 1 == 0:
                    cpt_in += n_in * fac_im1
                self.wfin_cpt = nn.Conv1d(cpt_in * in_channels, n_out * fac_i * out_channels, 1)
                self.wfin_cbdcst = nn.Linear(n_rc_inp * in_channels, n_out * fac_i * out_channels)
                set_init_(self.wfin_cpt, self.wfin_cbdcst)

                self.bias_fin_rc = nn.Linear(n_rc_inp * in_channels, n_out * out_channels)
            else:
                self.add_module(f"layer_{i}", nn.Conv2d(fac_i * in_channels, fac_i * out_channels, 1))
                self.add_module(f"layer_{i}_rc", nn.Linear(n_rc_inp * in_channels, fac_i * out_channels))

                fac_im1, fac_ip1 = filter_facs[i-1], filter_facs[i+1]
                row_in, col_in = (fac_im1 + fac_i + 1) * in_channels, (fac_ip1 + fac_i + 1) * in_channels
                if i == 1: row_in += n_in * filter_facs[0] * in_channels
                if i == self.L - 2: col_in += n_out * filter_facs[-1] * in_channels
                self.add_module(f"layer_{i}_r", nn.Conv1d(row_in, fac_i * out_channels, 1))
                self.add_module(f"layer_{i}_c", nn.Conv1d(col_in, fac_i * out_channels, 1))
                set_init_(
                    getattr(self, f"layer_{i}"),
                    getattr(self, f"layer_{i}_rc"),
                    getattr(self, f"layer_{i}_r"),
                    getattr(self, f"layer_{i}_c"),
                )

                self.add_module(f"bias_{i}", nn.Conv1d(col_in, out_channels, 1))
                self.add_module(f"bias_{i}_rc", nn.Linear(n_rc_inp * in_channels, out_channels))
                set_init_(getattr(self, f"bias_{i}"), getattr(self, f"bias_{i}_rc"))

    def forward(self, wsfeat: WeightSpaceFeatures):
        wsfeat = shape_wsfeat_symmetry(wsfeat, self.network_spec)
        weights, biases = wsfeat.weights, wsfeat.biases
        col_means = [w.mean(dim=-1) for w in weights]  # shapes: (B, C_in, nrow_i=ncol_i+1)
        row_means = [w.mean(dim=-2) for w in weights]  # shapes: (B, C_in, ncol_i=nrow_i-1)
        rc_means = [w.mean(dim=(-1, -2)) for w in weights]  # shapes: (B, C_in)
        bias_means = [b.mean(dim=-1) for b in biases]  # shapes: (B, C_in)
        rm0 = rearrange(row_means[0], "b c_in ncol -> b (c_in ncol)")
        cmL = rearrange(col_means[-1], "b c_in nrowL -> b (c_in nrowL)")
        final_bias = rearrange(biases[-1], "b c_in nrow -> b (c_in nrow)")
        # (B, C_in * (2 * L + n_in + 2 * n_out))
        rc_inp = torch.cat(rc_means + bias_means + [rm0, cmL, final_bias], dim=-1)

        out_weights, out_biases = [], []
        for i, (weight, bias) in enumerate(wsfeat):
            if i == 0:
                rpt = [rearrange(weight, "b c_in nrow ncol -> b (c_in ncol) nrow"), row_means[1], bias]
                if i+1 == self.L - 1:
                    rpt.append(rearrange(weights[-1], "b c_in n_out nrow -> b (c_in n_out) nrow"))
                rpt = torch.cat(rpt, dim=-2)  # repeat ptwise across rows
                z1 = self.w0_rpt(rpt)
                z2 = self.w0_rbdcst(rc_inp)[..., None]  # (b, c_out * ncol, 1)
                z = rearrange(z1 + z2, "b (c_out ncol) nrow -> b c_out nrow ncol", ncol=weight.shape[-1])
                u1 = self.bias_0(rpt)  # (B, C_out, nrow)
                u2 = self.bias_0_rc(rc_inp).unsqueeze(-1)  # (B, C_out, 1)
                u = u1 + u2
            elif i == self.L - 1:
                cpt = [rearrange(weight, "b c_in nrow ncol -> b (c_in nrow) ncol"), col_means[-2]]  # b c_in ncol 
                if i - 1 == 0:
                    cpt.append(rearrange(weights[0], "b c_in ncol n_in -> b (c_in n_in) ncol"))
                z1 = self.wfin_cpt(torch.cat(cpt, dim=-2))  # (B, C_out * nrow, ncol)
                z2 = self.wfin_cbdcst(rc_inp)[..., None]  # (b, c_out * nrow, 1)
                z = rearrange(z1 + z2, "b (c_out nrow) ncol -> b c_out nrow ncol", nrow=weight.shape[-2])
                u = rearrange(self.bias_fin_rc(rc_inp), "b (c_out nrow) -> b c_out nrow", nrow=weight.shape[-2])
            else:
                z1 = getattr(self, f"layer_{i}")(weight)  # (B, C_out, nrow, ncol)
                z2 = getattr(self, f"layer_{i}_rc")(rc_inp)[..., None, None]
                row_bdcst = [row_means[i], col_means[i-1], biases[i-1]]
                col_bdcst = [col_means[i], bias, row_means[i+1]]
                if i == 1:
                    row_bdcst.append(rearrange(weights[0], "b c ncol n_in -> b (c n_in) ncol"))
                if i == len(weights) - 2:
                    col_bdcst.append(rearrange(weights[-1], "b c n_out nrow -> b (c n_out) nrow"))
                row_bdcst = torch.cat(row_bdcst, dim=-2)  # (B, (3 + ?n_in) * C_in, ncol)
                col_bdcst = torch.cat(col_bdcst, dim=-2)  # (B, (3 + ?n_out) * C_in, nrow)
                z3 = getattr(self, f"layer_{i}_r")(row_bdcst).unsqueeze(-2)  # (B, C_out, 1, ncol)
                z4 = getattr(self, f"layer_{i}_c")(col_bdcst).unsqueeze(-1)  # (B, C_out, nrow, 1)
                z = z1 + z2 + z3 + z4
                u1 = getattr(self, f"bias_{i}")(col_bdcst)  # (B, C_out, nrow)
                u2 = getattr(self, f"bias_{i}_rc")(rc_inp).unsqueeze(-1)  # (B, C_out, 1)
                u = u1 + u2
            out_weights.append(z)
            out_biases.append(u)
        return unshape_wsfeat_symmetry(WeightSpaceFeatures(out_weights, out_biases), self.network_spec)

    def __repr__(self):
        return f"HNPLinear(in_channels={self.c_in}, out_channels={self.c_out}, n_in={self.n_in}, n_out={self.n_out}, num_layers={self.L})"