import os

import hydra
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import wandb
from tqdm import trange
import cv2
from PIL import Image
from omegaconf import OmegaConf

from nfn.common import network_spec_from_wsfeat, WeightSpaceFeatures
from experiments.data_utils import cycle, SirenAndOriginalDataset
from experiments.siren_utils import get_batch_siren, unprocess_img_arr


def params_to_func_params(params: WeightSpaceFeatures):
    """Convert our WeightSpaceFeatures object to a tuple of parameters for the functional model."""
    out_params = []
    for weight, bias in params:
        assert weight.shape[1] == bias.shape[1] == 1
        out_params.append(weight.squeeze(1))
        out_params.append(bias.squeeze(1))
    return tuple(out_params)


@torch.no_grad()
def evaluate(nfnet, loader, batch_siren):
    orig_state = nfnet.training
    nfnet.eval()
    recon_loss = 0
    tot_examples = 0
    for wts_and_bs, img, _ in loader:
        params = WeightSpaceFeatures(*wts_and_bs).to("cuda")
        img = img.cuda()
        delta = nfnet(params)
        new_params = params + delta
        func_params = params_to_func_params(new_params)
        outs = batch_siren(func_params)
        recon_loss += ((outs - img)**2).mean().item() * img.shape[0]
        tot_examples += img.shape[0]
    nfnet.train(orig_state)
    return recon_loss / tot_examples


def sharpen(img):
    kernel = np.array([[-.25,-.25,-.25], [-.25,3,-.25], [-.25,-.25,-.25]])
    img = cv2.filter2D(img, -1, kernel)
    return img


def inrease_contrast(img):
    # https://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(3,3))
    cl = clahe.apply(l_channel)
    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl,a,b))
    # Converting image from LAB Color model to BGR color space
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_img


@torch.no_grad()
def sample(nfnet, loader, batch_siren):
    orig_state = nfnet.training
    nfnet.eval()
    wts_and_bs, true_img, _ = next(iter(loader))
    params = WeightSpaceFeatures(*wts_and_bs).to("cuda")
    orig_outs = batch_siren(params_to_func_params(params))
    orig_outs = unprocess_img_arr(orig_outs.cpu().numpy())
    delta = nfnet(params)
    new_params = params_to_func_params(params + delta)
    new_outs = batch_siren(new_params)
    new_outs = unprocess_img_arr(new_outs.cpu().numpy())
    nfnet.train(orig_state)
    return orig_outs, new_outs, unprocess_img_arr(true_img.cpu().numpy())


def main(cfg):
    kernel = np.ones((3, 3), np.uint8)
    style_to_function = {
        'dilate': lambda im: cv2.dilate(im, kernel, iterations=1),
        'sharpen': sharpen,
        'contrast': inrease_contrast,
        'erode': lambda im: cv2.erode(im, np.ones((2, 2), np.uint8), iterations=1),
        'gradient': lambda im: cv2.morphologyEx(im, cv2.MORPH_GRADIENT, np.ones((2, 2), np.uint8)),
    }
    wandb.init(project=f"stylize_siren", reinit=True, config=OmegaConf.to_container(cfg, resolve=True))

    data_tfm = transforms.Compose([
        transforms.Lambda(np.array),
        transforms.Lambda(style_to_function[cfg.style]),
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    dset = SirenAndOriginalDataset(cfg.siren_path, "randinit_smaller", "./experiments/data", data_tfm)
    trainset, testset = Subset(dset, range(50_000)), Subset(dset, range(50_000, 60_000))
    trainset, valset = Subset(trainset, range(45_000)), Subset(trainset, range(45_000, 50_000))
    batch_siren = get_batch_siren(dset.data_type)
    trainloader = DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=8, drop_last=True)
    valloader = DataLoader(valset, batch_size=32, shuffle=False, num_workers=8, drop_last=True)

    spec = network_spec_from_wsfeat(WeightSpaceFeatures(*next(iter(trainloader))[0]).to("cpu"), set_all_dims=True)
    nfnet = hydra.utils.instantiate(cfg.nfnet, spec).cuda()
    print(nfnet)
    print(f"Total params in NFN: {sum(p.numel() for p in nfnet.parameters())}.")

    opt = hydra.utils.instantiate(cfg.opt, nfnet.parameters())
    sched = hydra.utils.call(cfg.sched, opt, cfg.max_steps)
    best_val_loss = float("inf")
    train_iter = cycle(trainloader)
    outer_pbar = trange(0, cfg.max_steps, position=0)
    for step in outer_pbar:
        wts_and_bs, img, _ = next(train_iter)
        params = WeightSpaceFeatures(*wts_and_bs).to("cuda")
        img = img.cuda()
        delta = nfnet(params)
        new_params = params + delta
        func_params = params_to_func_params(new_params)
        outs = batch_siren(func_params)
        opt.zero_grad()
        recon_loss = ((outs - img)**2).mean()
        recon_loss.backward()
        opt.step()
        sched.step()
        outer_pbar.set_description(f"recon_loss: {recon_loss.item():.3f}")
        if step % 10 == 0:
            wandb.log({
                "recon_loss/train": recon_loss.item(),
                "lr": opt.param_groups[0]["lr"],
            }, step=step)
        if step % 1000 == 0:
            val_recon_loss = evaluate(nfnet, valloader, batch_siren)
            orig_siren, new_siren, true_img = sample(nfnet, valloader, batch_siren)
            wandb.log({
                "recon_loss/val": val_recon_loss,
                "orig_siren_samples/val": [wandb.Image(Image.fromarray(x)) for x in orig_siren],
                "new_siren_samples/val": [wandb.Image(Image.fromarray(x)) for x in new_siren],
                "true_img_samples/val": [wandb.Image(Image.fromarray(x)) for x in true_img],
            }, step=step)
            if val_recon_loss < best_val_loss:
                torch.save(nfnet.state_dict(), os.path.join(cfg.output_dir, "best.pt"))
                best_val_loss = val_recon_loss
    outer_pbar.close()
    nfnet.load_state_dict(torch.load(os.path.join(cfg.output_dir, "best.pt")))
    testloader = DataLoader(testset, batch_size=cfg.batch_size, num_workers=8)
    test_recon_loss = evaluate(nfnet, testloader, batch_siren)
    orig_siren, new_siren, true_img = sample(nfnet, testloader, batch_siren)
    wandb.log({
        "recon_loss/test": test_recon_loss,
        "orig_siren_samples/test": [wandb.Image(Image.fromarray(x)) for x in orig_siren],
        "new_siren_samples/test": [wandb.Image(Image.fromarray(x)) for x in new_siren],
        "true_img_samples/test": [wandb.Image(Image.fromarray(x)) for x in true_img],
    }, step=step)
    wandb.finish()