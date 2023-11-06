from typing import Optional, Union, Type
import torch
from torch import nn
from einops.layers.torch import Rearrange

from nfn.layers import Pointwise, NPLinear, HNPLinear, FlattenWeights, NPPool
from nfn.layers import TupleOp, ResBlock, HNPPool, ParamLayerNorm, SimpleLayerNorm, ChannelDropout
from nfn.layers import StatFeaturizer, GaussianFourierFeatureTransform, IOSinusoidalEncoding, FlattenWeights
from nfn.layers import NPAttention
from nfn.common import NetworkSpec, WeightSpaceFeatures
from perceiver_pytorch import Perceiver


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
        self.sigmoid = sigmoid
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
        self.nfnet_features = nn.Sequential(*layers)
        self.head = head_cls(network_spec, prev_channels, append_stats)

    def forward(self, params):
        features = self.nfnet_features(self.preprocess(params))
        if self.stats is not None:
            features = torch.cat([features, self.stats(params)], dim=-1)
        return self.head(features)


class InvariantResNFN(nn.Module):
    """Invariant residual hypernetwork. Outputs a scalar."""
    def __init__(
        self,
        network_spec: NetworkSpec,
        hchannels,
        head_cls,
        mode="full",
        feature_dropout=0,
        inp_enc_cls=None,
        pos_enc_cls=None,
    ):
        super().__init__()
        self.normalize = False
        layers = []
        prev_channels = 1
        if inp_enc_cls is not None:
            inp_enc: GaussianFourierFeatureTransform = inp_enc_cls(network_spec, prev_channels)
            layers.append(inp_enc)
            prev_channels = 2 * inp_enc._mapping_size
        if pos_enc_cls:
            pos_enc: IOSinusoidalEncoding = pos_enc_cls(network_spec)
            layers.append(pos_enc)
            prev_channels = pos_enc.num_out_chan(prev_channels)
        for i, num_channels in enumerate(hchannels):
            hlayer = MODE2LAYER[mode](network_spec, in_channels=prev_channels, out_channels=num_channels)
            if i == 0:
                layers.extend([hlayer, TupleOp(nn.ReLU())])
                if feature_dropout > 0:
                    layers.append(TupleOp(nn.Dropout(p=feature_dropout)))
            elif i == len(hchannels) - 1:
                layers.extend([hlayer, TupleOp(nn.ReLU())])
                if feature_dropout > 0:
                    layers.append(TupleOp(nn.Dropout(p=feature_dropout)))
            else:
                assert num_channels == prev_channels
                norm = SimpleLayerNorm(network_spec, prev_channels)
                hlayer = ResBlock(hlayer, TupleOp(nn.ReLU()), feature_dropout, norm)
                layers.append(hlayer)
            prev_channels = num_channels
        self.features = nn.Sequential(*layers)
        self.head = head_cls(network_spec, prev_channels, append_stats=False)

    def forward(self, params):
        return self.head(self.features(params))


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


class Perceiver2d(nn.Module):
    def __init__(
        self,
        network_spec,
        in_channels,
        append_stats,
        num_classes,
        depth=1,
        self_per_cross_attn=2,
        num_latents=32,
        latent_dim=128,
        latent_heads=4,
        dropout=0.1,
        pool_latents=True,
    ):
        super().__init__()
        del append_stats
        self.flatten = FlattenWeights(network_spec)
        self.model = Perceiver(
            input_channels=in_channels,
            input_axis=1,
            num_freq_bands=6,
            max_freq=10.,
            depth=depth,
            num_latents=num_latents,
            latent_dim=latent_dim,
            cross_heads=1,
            latent_heads=latent_heads,
            cross_dim_head=64,
            latent_dim_head=64,
            attn_dropout=dropout,
            ff_dropout=dropout,
            weight_tie_layers=False,
            fourier_encode_data=False,
            self_per_cross_attn=self_per_cross_attn,
            final_classifier_head=pool_latents,
            num_classes=num_classes,
        )
        self.num_latents = num_latents
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.unflatten = nn.Unflatten(1, (int(num_latents**0.5), int(num_latents**0.5)))
        self.conv = nn.Sequential(
            nn.Conv2d(latent_dim, 64, 2), nn.ReLU(), nn.Dropout(p=dropout), # (b, 64, sqrt(n)-1, sqrt(n)-1)
            nn.Conv2d(64, 64, 2), nn.ReLU(), nn.Dropout(p=dropout), # (b, 64, sqrt(n)-2, sqrt(n)-2)
            nn.Flatten(),
            nn.Linear(64 * (int(num_latents**0.5-2))**2, num_classes) # (b, num_classes)
        ) if not pool_latents else nn.Identity()

    def forward(self, params):
        perceiver_out = self.model(self.flatten(params)) # (b, n, c)
        conv_input = self.unflatten(perceiver_out).permute(0, 3, 1, 2) # (b, c, sqrt(n), sqrt(n))
        return self.conv(conv_input)


class PerceiverNet(nn.Module):
    def __init__(
        self,
        network_spec,
        in_channels,
        append_stats,
        num_classes,
        depth=1,
        self_per_cross_attn=2,
        num_latents=32,
        latent_dim=128,
        latent_heads=4,
        latent_dim_head=64,
        cross_heads=1,
        cross_dim_head=64,
        dropout=0.1,
        pool_latents=True,
    ):
        super().__init__()
        del append_stats
        self.flatten = FlattenWeights(network_spec)
        self.model = Perceiver(
            input_channels=in_channels,
            input_axis=1,
            num_freq_bands=6,
            max_freq=10.,
            depth=depth,
            num_latents=num_latents,
            latent_dim=latent_dim,
            cross_heads=cross_heads,
            latent_heads=latent_heads,
            cross_dim_head=cross_dim_head,
            latent_dim_head=latent_dim_head,
            attn_dropout=dropout,
            ff_dropout=dropout,
            weight_tie_layers=False,
            fourier_encode_data=False,
            self_per_cross_attn=self_per_cross_attn,
            final_classifier_head=pool_latents,
            num_classes=num_classes,
        )
        self.mlp = nn.Sequential(
            Rearrange("b n c -> b (n c)"),
            nn.Linear(latent_dim * num_latents, 1000), nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(1000, 1000), nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(1000, num_classes),
        ) if not pool_latents else nn.Identity()

    def forward(self, params):
        perceiver_out = self.model(self.flatten(params))
        return self.mlp(perceiver_out)
    

class MlpNFN(NormalizingModule):
    """Hypernetwork trained with weight permutation augmentations. Outputs a scalar."""
    def __init__(
        self,
        network_spec: NetworkSpec,
        h_size,
        num_layers=3,
        dropout=0.0,
        sigmoid=False,
        normalize=False,
    ):
        super().__init__(normalize=normalize)
        activations = [nn.Sigmoid()] if sigmoid else []
        hidden_layers = []
        for _ in range(num_layers - 2):
            hidden_layers.append(nn.Linear(h_size, h_size))
            hidden_layers.append(nn.ReLU())
            hidden_layers.append(nn.Dropout(p=dropout))
        self.hypernetwork = nn.Sequential(
            FlattenWeights(network_spec),
            nn.Flatten(start_dim=-2),
            nn.Linear(network_spec.get_num_params(), h_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            *hidden_layers,
            nn.Linear(h_size, 10),
            *activations,
        )

    def forward(self, params):
        return self.hypernetwork(self.preprocess(params))


class Block(nn.Module):
    def __init__(
        self,
        network_spec,
        channels,
        ff_factor=2,
        num_heads=8,
        dropout=0.1,
        share_projections=True,
        # These two are for ablations only, should always be False otherwise.
        ablate_crossterm=False,
        ablate_diagonalterm=False,
    ):
        super().__init__()
        self.ln1 = SimpleLayerNorm(network_spec, channels)
        self.ln2 = SimpleLayerNorm(network_spec, channels)
        self.attn = NPAttention(
            network_spec,
            channels,
            num_heads,
            dropout,
            share_projections=share_projections,
            ablate_crossterm=ablate_crossterm,
            ablate_diagonalterm=ablate_diagonalterm,
        )
        self.drop = TupleOp(nn.Dropout(dropout))
        self.ff = nn.Sequential(
            Pointwise(network_spec, channels, ff_factor * channels),
            TupleOp(nn.GELU()),
            Pointwise(network_spec, ff_factor * channels, channels),
            TupleOp(nn.Dropout(dropout)),
        )

    def forward(self, x):
        x = x + self.drop(self.attn(self.ln1(x)))
        return x + self.ff(self.ln2(x))