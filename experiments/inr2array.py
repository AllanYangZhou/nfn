import os
import math
from PIL import Image
import wandb
from omegaconf import OmegaConf
import torch
from torch import nn
import torch._dynamo as dynamo
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from einops.layers.torch import Reduce, Rearrange
from tqdm import trange
import hydra
from nfn.layers import LearnedPosEmbedding, ResBlock, NPLinear, HNPLinear, TupleOp, TupleOp, NPPool
from nfn.layers import GaussianFourierFeatureTransform, SimpleLayerNorm, FlattenWeights, UnflattenWeights
from nfn.common import network_spec_from_wsfeat, WeightSpaceFeatures, NetworkSpec, params_to_func_params
from experiments.data_utils import cycle, SirenAndOriginalDataset, AlignedSampler
from experiments.train_utils import get_linear_warmup_with_cos_decay
from experiments.models import Block
from experiments.siren_utils import get_batch_siren, get_spatial_batch_siren, unprocess_img_arr
from experiments.siren_utils import HyperNetwork, SimpleHyperNetwork
from perceiver_pytorch import Perceiver


def simple_npblock(network_spec, n_chan):
    return nn.Sequential(NPLinear(network_spec, n_chan, n_chan), TupleOp(nn.ReLU()))


def np_resblock(network_spec, n_chan, dropout=0.1):
    return ResBlock(
        NPLinear(network_spec, n_chan, n_chan),
        TupleOp(nn.ReLU()), dropout=dropout,
        norm=SimpleLayerNorm(network_spec, n_chan),
    )


def hnp_resblock(network_spec, n_chan, dropout=0.1):
    return ResBlock(
        HNPLinear(network_spec, n_chan, n_chan),
        TupleOp(nn.ReLU()), dropout=dropout,
        norm=SimpleLayerNorm(network_spec, n_chan),
    )


class PerceiverPooling(nn.Module):
    def __init__(self, network_spec, n_chan, n_latent=64, latent_dim=256, reduce=True, attn_dropout=0.1, ff_dropout=0.1, self_per_cross_attn=2):
        super().__init__()
        self.reduce, self.n_latent = reduce, n_latent
        self.net = nn.Sequential(
            FlattenWeights(network_spec),
            Perceiver(
                input_channels=n_chan,
                input_axis=1,
                depth=1,  # number of cross-attns
                num_latents=n_latent,  # number of latent vectors
                latent_dim=latent_dim,  # dimension of latent vectors.
                cross_heads=1,  # number of cross-attention heads
                latent_heads=8,  #  number of latent self-attention heads
                cross_dim_head=n_chan,  # per head dim of cross-attention
                latent_dim_head=64,  # per head dim of self attention
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                weight_tie_layers=False,
                fourier_encode_data=False,
                self_per_cross_attn=self_per_cross_attn,  # number of self-attns per cross-attn
                final_classifier_head=False,
            ),
            Reduce("b n c -> b c", reduction="mean") if reduce else nn.Identity(),
        )
        self.latent_dim = latent_dim

    def forward(self, x):
        return self.net(x)

    def get_feature_shape(self):
        return (self.latent_dim,) if self.reduce else (self.n_latent, self.latent_dim)

class NPPooling(nn.Module):
    def __init__(self, network_spec, n_chan, reduce=True, num_patches=64, latent_dim=64):
        super().__init__()
        """Assumes the input to the pooling layer has n_chan channels.
            If reduce is True, the output will be a single vector of size latent_dim.
            If reduce is False, the output will be a list of num_patches vectors, each of size latent_dim.
        """
        self.reduce, self.latent_dim = reduce, latent_dim
        self.n_chan, self.num_patches = n_chan, num_patches
        num_pool_outs = NPPool.get_num_outs(network_spec)
        num_latent = latent_dim if reduce else num_patches * latent_dim
        self.pool = nn.Sequential(
            NPPool(network_spec), nn.Flatten(-2),
            nn.Linear(num_pool_outs * n_chan, 1000), nn.ReLU(),
            nn.Linear(1000, num_latent)
        )
        self.reduce_fn = nn.Identity()
        if not reduce:
            self.reduce_fn = Rearrange("b (l d) -> b l d", l=num_patches, d=latent_dim)

    def forward(self, x):
        return self.reduce_fn(self.pool(x))

    def get_feature_shape(self):
        return (self.latent_dim,) if self.reduce else (self.num_patches, self.latent_dim)


BLOCK_TYPES = {
    "np": simple_npblock,
    "np_residual": np_resblock,
    "hnp_residual": hnp_resblock,
    "nft": Block,
}
DEC_TYPES = {"sitzmann": HyperNetwork, "simple": SimpleHyperNetwork}
class AutoEncoder(nn.Module):
    def __init__(
        self, network_spec: NetworkSpec, dset_data_type,
        block_type="nft", pool_cls=PerceiverPooling,
        num_blocks=3, spatial=False,
        compile=False,
        enc_scale=3, enc_map_size=128,  # scalen and dimension of gaussian fourier features
        dec_type="sitzmann",  # sitzmann: one MLP to per each SIREN weight/bias. simple: one MLP total.
        dec_hidden_size=256, dec_hidden_layers=1,
        additive=False,  # if true, the decoder output is a delta, added to a learned set of SIREN parameters
        debug_compile=False,  # if true, explains compilation process on first forward
        **block_kwargs,
    ):
        super().__init__()
        self.network_spec = network_spec
        self.spatial = spatial
        self.compile, self.debug_compile = compile, debug_compile
        n_chan = 2 * enc_map_size
        self.encoder = nn.Sequential(
            GaussianFourierFeatureTransform(network_spec, 1, mapping_size=enc_map_size, scale=enc_scale),
            LearnedPosEmbedding(network_spec, n_chan),
            *[BLOCK_TYPES[block_type](network_spec, n_chan, **block_kwargs) for _ in range(num_blocks)],
            pool_cls(network_spec, n_chan, reduce=not spatial),
        )
        self.latent_shape = self.encoder[-1].get_feature_shape()
        self.decoder = nn.Sequential(
            DEC_TYPES[dec_type](network_spec, self.latent_shape[-1], dec_hidden_size, dec_hidden_layers),
            Rearrange("b l -> b l ()"),
            UnflattenWeights(network_spec),
        )
        if spatial:
            assert len(self.latent_shape) == 2
            self.batch_siren, init_params = get_spatial_batch_siren(dset_data_type)
        else:
            self.batch_siren, init_params = get_batch_siren(dset_data_type)
        self.init_params = None
        if additive:
            self.init_params = nn.ParameterList([nn.Parameter(p) for p in init_params])
            # Empirical observation: with these scales, the additive deltas will be ~1/10 the size of
            # corresponding init_params.
            additive_scales = []
            for p in self.init_params:  # if numel==1, std is nan.
                if p.numel() == 1: additive_scales.append(nn.Parameter(p.detach().abs().mean()))
                else: additive_scales.append(nn.Parameter(p.detach().std().requires_grad_()))
            self.additive_scales = nn.ParameterList(additive_scales)
        if self.compile: self._fast_forward = torch.compile(self._forward_helper)

    def encode(self, x):
        return self.encoder(x)

    def _forward_helper(self, x):
        """Functorch prevents compiling the entire forward method, but this helper can be compiled."""
        z = self.encoder(x)
        if self.spatial:
            bs, num_latents = z.shape[:2]
            z = torch.flatten(z, end_dim=1)
        out = self.decoder(z)
        if self.spatial:
            def unflatten(arr):
                # remove singleton channel dim
                return arr.reshape(bs, num_latents, *arr.shape[1:]).squeeze(2)
            out = out.map(unflatten)
        return out

    def forward(self, x):
        if self.debug_compile:
            _, _, _, _, _, explanation_verbose = dynamo.explain(self._forward_helper, x)
            print(explanation_verbose)
            self.debug_compile = False
        out = self._fast_forward(x) if self.compile else self._forward_helper(x)
        out = params_to_func_params(out)
        if self.init_params is not None:
            out = [p * scale + init_p[None, None] for p, init_p, scale in zip(out, self.init_params, self.additive_scales)]
        out = self.batch_siren(out)
        return out

    def sample(self, num_samples):
        device = next(self.parameters()).device
        z = torch.randn(num_samples, *self.latent_shape).to(device)
        if self.spatial:
            _, num_latents = z.shape[:2]
            z = torch.flatten(z, end_dim=1)
        out = self.decoder(z)
        if self.spatial:
            def unflatten(arr):
                # remove singleton channel dim
                return arr.reshape(num_samples, num_latents, *arr.shape[1:]).squeeze(2)
            out = out.map(unflatten)
        out = params_to_func_params(out)
        if self.init_params is not None:
            out = [p * scale + init_p[None, None] for p, init_p, scale in zip(out, self.init_params, self.additive_scales)]
        out = self.batch_siren(out)
        return out


@torch.no_grad()
def evaluate(nfnet, loader, orig_batch_siren, true_target, loss_type='l2'):
    orig_state = nfnet.training
    nfnet.eval()
    loss = 0
    for wts_and_bs, img, _ in loader:
        params = WeightSpaceFeatures(*wts_and_bs).to("cuda")
        if not true_target:
            img = orig_batch_siren(params_to_func_params(params))
        pred_img = nfnet(params)
        batch_loss = compute_losses(img.cuda(), pred_img, loss_type)
        loss += batch_loss
    nfnet.train(orig_state)
    return loss / len(loader)


@torch.no_grad()
def sample(nfnet):
    orig_state = nfnet.training
    nfnet.eval()
    samples = nfnet.sample(32)
    nfnet.train(orig_state)
    return unprocess_img_arr(samples.cpu().numpy())


@torch.no_grad()
def sample_recon(nfnet, loader, orig_batch_siren):
    orig_state = nfnet.training
    nfnet.eval()
    wts_and_bs, true_img, _ = next(iter(loader))
    params = WeightSpaceFeatures(*wts_and_bs).to("cuda")
    params, true_img = params.map(lambda x: x[:32]), true_img[:32]
    orig_outs = orig_batch_siren(params_to_func_params(params))
    orig_outs = unprocess_img_arr(orig_outs.cpu().numpy())
    new_outs = nfnet(params)
    new_outs = unprocess_img_arr(new_outs.cpu().numpy())
    nfnet.train(orig_state)
    return orig_outs, new_outs, unprocess_img_arr(true_img.cpu().numpy())


def compute_losses(x, x_recon, loss_type='l2'):
    if loss_type == 'l2':
        return F.mse_loss(x_recon, x)
    else:
        return F.l1_loss(x_recon, x)


def train_and_eval(cfg):
    ckpt_path = os.path.join(cfg.output_dir, "checkpoint.pt")
    wandb_id, prev_ckpt = None, None
    if os.path.exists(ckpt_path):
        print("Resuming from checkpoint.")
        prev_ckpt = torch.load(ckpt_path)
        wandb_id = prev_ckpt["wandb_run_id"]
    if not cfg.debug:
        wandb.init(
            project="inr2array",
            reinit=True,
            resume="must" if wandb_id is not None else False,
            id=wandb_id,
        )
        if prev_ckpt is None:
            cfg_dict = OmegaConf.to_container(cfg, resolve=True)
            wandb.config.update(cfg_dict)
    trainset: SirenAndOriginalDataset = hydra.utils.instantiate(cfg.dset, split="train")
    valset: SirenAndOriginalDataset = hydra.utils.instantiate(cfg.dset, split="val")
    if cfg.extra_aug > 0:
        aug_dsets = []
        for i in range(cfg.extra_aug):
            aug_dsets.append(hydra.utils.instantiate(cfg.dset, siren_prefix=f"randinit_smaller_aug{i}", split="train"))
        trainset = ConcatDataset([trainset] + aug_dsets)
    print(f"Dataset sizes: train={len(trainset)}, val={len(valset)}.")
    sampler = None
    if cfg.aligned_sampling:
        assert cfg.extra_aug > 0, "Aligned sampling only makes sense with extra augmentations."
        sampler = AlignedSampler(trainset, len(trainset) // (cfg.extra_aug + 1))
    trainloader = DataLoader(trainset, batch_size=cfg.bs, shuffle=sampler is None, num_workers=8, drop_last=True, sampler=sampler)
    valloader = DataLoader(valset, batch_size=cfg.bs, shuffle=False, num_workers=8, drop_last=True)

    spec = network_spec_from_wsfeat(WeightSpaceFeatures(*next(iter(trainloader))[0]).to("cpu"), set_all_dims=True)
    nfnet: AutoEncoder = hydra.utils.instantiate(cfg.model, spec, valset.data_type).to("cuda")
    nfnet_fast = torch.compile(nfnet) if cfg.compile else nfnet
    if not cfg.debug and cfg.watch_grads: wandb.watch(nfnet_fast, log="gradients", log_freq=1000)
    orig_batch_siren = get_batch_siren(valset.data_type)[0]
    print(nfnet)
    print(f"Total params in NFN: {sum(p.numel() for p in nfnet.parameters())}.")

    opt = torch.optim.AdamW(nfnet.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = get_linear_warmup_with_cos_decay(opt, total_steps=cfg.total_steps, warmup_steps=cfg.warmup_steps, decay_start=cfg.decay_start)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp_enabled)
    start_step, best_val_loss = 0, float("inf")

    if prev_ckpt is not None:
        nfnet.load_state_dict(prev_ckpt["nfnet"])
        opt.load_state_dict(prev_ckpt["opt"])
        start_step = prev_ckpt["step"]
        best_val_loss = prev_ckpt["best_val_loss"]
        scheduler.load_state_dict(prev_ckpt["scheduler"])
        scaler.load_state_dict(prev_ckpt["scaler"])

    train_iter = cycle(trainloader)
    outer_pbar = trange(start_step, cfg.total_steps, position=0)
    for step in outer_pbar:
        wts_and_bs, img, _ = next(train_iter)
        params = WeightSpaceFeatures(*wts_and_bs).to("cuda")
        if not cfg.true_target:
            img = orig_batch_siren(params_to_func_params(params))
        if step == 0 and cfg.debug_compile:
            _, _, _, _, _, explanation_verbose = dynamo.explain(nfnet, params)
            print(explanation_verbose)
        opt.zero_grad()
        amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[cfg.amp_dtype]
        with torch.amp.autocast("cuda", enabled=cfg.amp_enabled, dtype=amp_dtype):
            pred_img = nfnet_fast(params)
            train_loss = compute_losses(img.cuda(), pred_img)
        scaler.scale(train_loss).backward()
        scaler.unscale_(opt)
        if cfg.grad_clip is not None:
            tot_norm = nn.utils.clip_grad_norm_(nfnet.parameters(), cfg.grad_clip)
        scaler.step(opt)
        scaler.update()
        scheduler.step()
        if step % 10 == 0:
            if not cfg.debug:
                wandb.log({
                    "train/loss": train_loss.item(),
                    "lr": scheduler.get_last_lr()[0],
                    "tot_norm": tot_norm.item() if cfg.grad_clip is not None else 0,
                }, step=step)
        if step % 1000 == 0:
            val_loss = evaluate(nfnet, valloader, orig_batch_siren, cfg.true_target, cfg.loss_type)
            orig_siren, new_siren, true_img = sample_recon(nfnet, valloader, orig_batch_siren)
            metrics = {
                "val/loss": val_loss.item(),
                "val/orig_siren": [wandb.Image(Image.fromarray(x)) for x in orig_siren],
                "val/recon_siren": [wandb.Image(Image.fromarray(x)) for x in new_siren],
                "val/true_img": [wandb.Image(Image.fromarray(x)) for x in true_img],
            }
            if not cfg.debug: wandb.log(metrics, step=step)
            if val_loss < best_val_loss:
                torch.save(nfnet.state_dict(), os.path.join(cfg.output_dir, "best_nfnet.pt"))
                best_val_loss = val_loss
            torch.save({
                "step": step,
                "nfnet": nfnet.state_dict(),
                "opt": opt.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_val_loss": best_val_loss,
                "wandb_run_id": wandb.run.id if not cfg.debug else None,
            }, ckpt_path)
        outer_pbar.set_description(f"train_loss: {train_loss.item():.3f}, val_loss: {val_loss.item():.3f}")
        if cfg.debug: break
    print("Testing")
    nfnet.load_state_dict(torch.load(os.path.join(cfg.output_dir, "best_nfnet.pt")))
    testset: SirenAndOriginalDataset = hydra.utils.instantiate(cfg.dset, split="test")
    testloader = DataLoader(testset, batch_size=cfg.bs, shuffle=False, num_workers=8, drop_last=True)
    test_loss = evaluate(nfnet, testloader, orig_batch_siren, cfg.true_target, cfg.loss_type)
    trainset: SirenAndOriginalDataset = hydra.utils.instantiate(cfg.dset, split="train")
    trainloader = DataLoader(trainset, batch_size=cfg.bs, shuffle=False, num_workers=8, drop_last=True)
    train_loss = evaluate(nfnet, trainloader, orig_batch_siren, cfg.true_target, cfg.loss_type)
    print(f"Test loss: {test_loss.item():.3f}")
    if not cfg.debug: wandb.log({"test/loss": test_loss.item(), "train/final_loss": train_loss.item()}, step=cfg.total_steps)
