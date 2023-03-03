from typing import Optional, Union, Type
import torch
from torch import nn

from nfn.layers import Pointwise, NPLinear, HNPLinear, NPPool
from nfn.layers import TupleOp, HNPPool, ParamLayerNorm, SimpleLayerNorm, ChannelDropout
from nfn.layers import StatFeaturizer, GaussianFourierFeatureTransform, IOSinusoidalEncoding
from nfn.common import NetworkSpec, WeightSpaceFeatures


MODE2LAYER = {
    "PT": Pointwise,
    "NP": NPLinear,
    "NP-PosEmb": lambda *args, **kwargs: NPLinear(*args, io_embed=True, **kwargs),
    "HNP": HNPLinear,
}

LN_DICT = {
    "param": ParamLayerNorm,
    "simple": SimpleLayerNorm,
}

POOL_DICT = {"HNP": HNPPool, "NP": NPPool}


class NormalizingModule(nn.Module):
    def __init__(self, normalize=False):
        super().__init__()
        self.normalize = normalize

    def set_stats(self, mean_std_stats):
        if self.normalize:
            print("Setting stats")
            weight_stats, bias_stats = mean_std_stats
            for i, (w, b) in enumerate(zip(weight_stats, bias_stats)):
                mean_weights, std_weights = w
                mean_bias, std_bias = b
                # wherever std_weights < 1e-5, set to 1
                std_weights = torch.where(std_weights < 1e-5, torch.ones_like(std_weights), std_weights)
                std_bias = torch.where(std_bias < 1e-5, torch.ones_like(std_bias), std_bias)
                self.register_buffer(f"mean_weights_{i}", mean_weights)
                self.register_buffer(f"std_weights_{i}", std_weights)
                self.register_buffer(f"mean_bias_{i}", mean_bias)
                self.register_buffer(f"std_bias_{i}", std_bias)

    def _normalize(self, params):
        out_weights, out_bias = [], []
        for i, (w, b) in enumerate(params):
            mean_weights_i, std_weights_i = getattr(self, f"mean_weights_{i}"), getattr(self, f"std_weights_{i}")
            mean_bias_i, std_bias_i = getattr(self, f"mean_bias_{i}"), getattr(self, f"std_bias_{i}")
            out_weights.append((w - mean_weights_i) / std_weights_i)
            out_bias.append((b - mean_bias_i) / std_bias_i)
        return WeightSpaceFeatures(out_weights, out_bias)


    def preprocess(self, params):
        if self.normalize:
            params = self._normalize(params)
        return params


class MlpHead(nn.Module):
    def __init__(
        self,
        network_spec,
        in_channels,
        append_stats,
        num_out=1,
        h_size=1000,
        dropout=0.0,
        lnorm=False,
        pool_mode="HNP",
        sigmoid=False
    ):
        super().__init__()
        head_layers = []
        pool_cls = POOL_DICT[pool_mode]
        head_layers.extend([pool_cls(network_spec), nn.Flatten(start_dim=-2)])
        num_pooled_outs = in_channels * pool_cls.get_num_outs(network_spec) + StatFeaturizer.get_num_outs(network_spec) * int(append_stats)
        head_layers.append(nn.Linear(num_pooled_outs, h_size))
        for i in range(2):
            if lnorm:
                head_layers.append(nn.LayerNorm(h_size))
            head_layers.append(nn.ReLU())
            if dropout > 0:
                head_layers.append(nn.Dropout(p=dropout))
            head_layers.append(nn.Linear(h_size, h_size if i == 0 else num_out))
        if sigmoid:
            head_layers.append(nn.Sigmoid())
        self.head = nn.Sequential(*head_layers)

    def forward(self, x):
        return self.head(x)


InpEncTypes = Optional[Union[Type[GaussianFourierFeatureTransform], Type[Pointwise]]]
class InvariantNFN(NormalizingModule):
    """Invariant hypernetwork. Outputs a scalar."""
    def __init__(
        self,
        network_spec: NetworkSpec,
        hchannels,
        head_cls,
        mode="HNP",
        feature_dropout=0,
        normalize=False,
        lnorm=None,
        append_stats=False,
        inp_enc_cls: InpEncTypes=None,
        pos_enc_cls: Optional[Type[IOSinusoidalEncoding]]=None,
        in_channels=1,
    ):
        super().__init__(normalize=normalize)
        self.stats = None
        if append_stats:
            self.stats = nn.Sequential(StatFeaturizer(), nn.Flatten(start_dim=-2))
        layers = []
        prev_channels = in_channels
        if inp_enc_cls is not None:
            inp_enc = inp_enc_cls(network_spec, in_channels)
            layers.append(inp_enc)
            prev_channels = inp_enc.out_channels
        if pos_enc_cls:
            pos_enc: IOSinusoidalEncoding = pos_enc_cls(network_spec)
            layers.append(pos_enc)
            prev_channels = pos_enc.num_out_chan(prev_channels)
        for num_channels in hchannels:
            layers.append(MODE2LAYER[mode](network_spec, in_channels=prev_channels, out_channels=num_channels))
            if lnorm is not None:
                layers.append(LN_DICT[lnorm](network_spec, num_channels))
            layers.append(TupleOp(nn.ReLU()))
            if feature_dropout > 0:
                layers.append(ChannelDropout(feature_dropout))
            prev_channels = num_channels
        self.hnet_features = nn.Sequential(*layers)
        self.head = head_cls(network_spec, prev_channels, append_stats)

    def forward(self, params):
        features = self.hnet_features(self.preprocess(params))
        if self.stats is not None:
            features = torch.cat([features, self.stats(params)], dim=-1)
        return self.head(features)


class StatNet(NormalizingModule):
    """Outputs a scalar."""
    def __init__(
        self,
        network_spec: NetworkSpec,
        h_size,
        dropout=0.0,
        sigmoid=False,
        normalize=False,
    ):
        super().__init__(normalize=normalize)
        activations = [nn.Sigmoid()] if sigmoid else []
        self.hypernetwork = nn.Sequential(
            StatFeaturizer(),
            nn.Flatten(start_dim=-2),
            nn.Linear(StatFeaturizer.get_num_outs(network_spec), h_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(h_size, h_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(h_size, 1),
            *activations
        )

    def forward(self, params):
        return self.hypernetwork(self.preprocess(params))