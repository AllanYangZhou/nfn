import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils import data
from tqdm import tqdm, trange
from omegaconf import OmegaConf
import hydra
import wandb

from experiments.data_utils import SirenDataset
from nfn.common import network_spec_from_wsfeat, WeightSpaceFeatures
from experiments.data_utils import compute_mean_std, cycle

from experiments.models import InvariantNFN

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def evaluate(nfnet, loader):
    orig_state = nfnet.training
    nfnet.eval()
    pbar = tqdm(loader, position=1, leave=False)
    labels, preds = [], []
    val_losses = []
    for wts_and_bs, label in pbar:
        params = WeightSpaceFeatures(*wts_and_bs).to("cuda")
        with torch.no_grad():
            pred = nfnet(params)
        loss = F.cross_entropy(pred, label.cuda())
        pbar.set_description(f"val loss={loss.item():.3f}")
        preds.append(torch.argmax(pred, -1).cpu().numpy())
        labels.append(label.numpy())
        val_losses.append(loss.item())
    pbar.close()
    val_acc = (np.concatenate(preds) == np.concatenate(labels)).mean().item()
    nfnet.train(orig_state)
    return val_acc, np.mean(val_losses)


def train_step(nfnet, opt, params, label):
    opt.zero_grad(True)
    pred = nfnet(params)
    loss = F.cross_entropy(pred, label.cuda())
    loss.backward()
    opt.step()
    return loss, pred


def main(cfg):
    prev_ckpt, wandb_id = None, None
    ckpt_path = os.path.join(cfg.output_dir, "checkpoint.pt")
    if os.path.exists(ckpt_path):
        print("Resuming from checkpoint.")
        prev_ckpt = torch.load(ckpt_path)
        wandb_id = prev_ckpt["wandb_run_id"]
    dset_name = os.path.basename(cfg.dset.data_path)
    wandb.init(project=f"classify_{dset_name}", reinit=True, resume="must" if wandb_id else False, id=wandb_id)
    if prev_ckpt is None:
        wandb.config.update(OmegaConf.to_container(cfg, resolve=True))
    trainset: SirenDataset = hydra.utils.instantiate(cfg.dset, split="train")
    valset: SirenDataset = hydra.utils.instantiate(cfg.dset, split="val")
    print(f"Train set and val set with {len(trainset)} and {len(valset)} examples.")
    if cfg.extra_aug > 0:
        aug_dsets = []
        for i in range(cfg.extra_aug):
            aug_dsets.append(hydra.utils.instantiate(cfg.dset, prefix=cfg.dset.prefix + f"_aug{i}", split="train"))
        trainset = data.ConcatDataset([trainset] + aug_dsets)
        print(f"Augmented training set with {len(trainset)} examples.")
    trainloader = data.DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=8, drop_last=True)
    valloader = data.DataLoader(valset, batch_size=cfg.batch_size, num_workers=8, drop_last=True)
    spec = network_spec_from_wsfeat(WeightSpaceFeatures(*next(iter(trainloader))[0]).to("cpu"), set_all_dims=True)
    nfnet: InvariantNFN = hydra.utils.instantiate(cfg.nfnet, spec)
    print(f"Total params in nfnet: {sum(p.numel() for p in nfnet.parameters())}.")
    if nfnet.normalize:
        nfnet.set_stats(compute_mean_std(trainloader, max_batches=(5_000 // cfg.batch_size) + 1))
    print(nfnet)
    nfnet.cuda()
    if cfg.compile: nfnet = torch.compile(nfnet)
    opt = hydra.utils.instantiate(cfg.opt, nfnet.parameters())
    sched = None
    if cfg.warmup_steps > 0:
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1., i / cfg.warmup_steps))
    start_step, best_val_acc = 0, 0

    if os.path.exists(ckpt_path):
        nfnet.load_state_dict(prev_ckpt["nfnet"])
        opt.load_state_dict(prev_ckpt["opt"])
        start_step = prev_ckpt["step"]
        best_val_acc = prev_ckpt["best_val_acc"]
        if sched is not None:
            sched.load_state_dict(prev_ckpt["sched"])

    train_iter = cycle(trainloader)
    outer_pbar = trange(start_step, cfg.max_steps, position=0)
    for step in outer_pbar:
        if step % 3000 == 0 or step == cfg.max_steps - 1:
            val_acc, val_loss = evaluate(nfnet, valloader)
            wandb.log({"val/acc": val_acc, "val/loss": val_loss}, step=step)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(nfnet.state_dict(), os.path.join(cfg.output_dir, "best.pt"))
            torch.save({
                "step": step,
                "nfnet": nfnet.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "best_val_acc": best_val_acc,
                "wandb_run_id": wandb.run.id,
            }, ckpt_path)
        wts_and_bs, label = next(train_iter)
        params = WeightSpaceFeatures(*wts_and_bs).to("cuda")
        loss, pred = train_step(nfnet, opt, params, label)
        if sched is not None:
            sched.step()
        outer_pbar.set_description(f"train_loss={loss.item():.3f},val_acc={val_acc:.3f}")
        if step % 10 == 0:
            metrics = {"train/loss": loss.item()}
            metrics["train/acc"] = (torch.argmax(pred.detach(), -1).cpu().numpy() == label.numpy()).mean().item()
            if sched is not None:
                metrics["train/lr"] = sched.get_last_lr()[0]
            wandb.log(metrics, step=step)
    outer_pbar.close()
    # load best.pt
    nfnet.load_state_dict(torch.load(os.path.join(cfg.output_dir, "best.pt")))
    testset = hydra.utils.instantiate(cfg.dset, split="test")
    print(f"Test set with {len(testset)} examples.")
    testloader = data.DataLoader(testset, batch_size=cfg.batch_size, num_workers=8)
    test_acc, test_loss = evaluate(nfnet, testloader)
    print(f"Test accuracy: {test_acc:.3f}, test loss: {test_loss:.3f}.")
    wandb.log({"test/acc": test_acc, "test/loss": test_loss}, step=step)
    
    final_train_acc, final_train_loss = evaluate(nfnet, trainloader)
    print(f"Final train accuracy: {final_train_acc:.3f}, final train loss: {final_train_loss:.3f}.")
    wandb.log({"final_train/acc": final_train_acc, "final_train/loss": final_train_loss}, step=step)
    wandb.finish()