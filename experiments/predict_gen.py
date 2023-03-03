import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
import os
from torch import nn
from torch.utils.data import DataLoader
import wandb
from sklearn.metrics import r2_score
from scipy.stats import kendalltau
from tqdm import tqdm

from nfn.common import network_spec_from_wsfeat, WeightSpaceFeatures
from experiments.data_utils import compute_mean_std
from experiments.models import NormalizingModule


@torch.no_grad()
def evaluate(nfnet, loader, loss_fn):
    nfnet.eval()
    pred, actual = [], []
    err, losses = [], []
    for wts_and_bs, acc in loader:
        acc, params = acc.float().cuda(), WeightSpaceFeatures(*wts_and_bs).to("cuda")
        pred_acc = nfnet(params).squeeze(-1)
        err.append(torch.abs(pred_acc - acc).mean().item())
        losses.append(loss_fn(pred_acc, acc).item())
        pred.append(pred_acc.detach().cpu().numpy())
        actual.append(acc.cpu().numpy())
    avg_err, avg_loss = np.mean(err), np.mean(losses)
    actual, pred = np.concatenate(actual), np.concatenate(pred)
    rsq = r2_score(actual, pred)
    tau = kendalltau(actual, pred).correlation  # NOTE: on newer scipy this is called "statistic"
    return avg_err, avg_loss, rsq, tau, actual, pred


def train(cfg):
    wandb.init(project="predict_gen", config=OmegaConf.to_container(cfg, resolve=True), reinit=True)
    trainset = hydra.utils.instantiate(cfg.dset, mode="train")
    valset = hydra.utils.instantiate(cfg.dset, mode="val")
    if cfg.debug:  # 2 batches for debugging
        trainset = torch.utils.data.Subset(trainset, range(2 * cfg.batch_size))
        valset = torch.utils.data.Subset(valset, range(2 * cfg.batch_size))
    print(f"Trainset size: {len(trainset)}, valset size: {len(valset)}.")
    trainloader = DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=8)
    valloader = DataLoader(valset, batch_size=cfg.batch_size, shuffle=False, num_workers=8)
    network_spec = network_spec_from_wsfeat(WeightSpaceFeatures(*next(iter(trainloader))[0]).to("cpu"))

    nfnet: NormalizingModule = hydra.utils.instantiate(cfg.nfnet, network_spec)
    print(nfnet)
    print(f"Total params in NFN: {sum(p.numel() for p in nfnet.parameters())}.")
    if nfnet.normalize:
        max_batches = 10 if cfg.debug else None
        nfnet.set_stats(compute_mean_std(trainloader, max_batches))
    nfnet.cuda()
    opt = torch.optim.Adam(nfnet.parameters(), lr=cfg.lr)
    sched = None
    if cfg.warmup_steps > 0:
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1., i / cfg.warmup_steps))

    loss_fn = {"mse": nn.MSELoss(), "bce": nn.BCELoss()}[cfg.loss_fn]
    step = 0
    best_rsq, best_tau = -float('inf'), -float('inf')
    for epoch in range(cfg.epochs):
        nfnet.train()
        for wts_and_bs, acc in tqdm(trainloader):
            acc, params = acc.float().cuda(), WeightSpaceFeatures(*wts_and_bs).to("cuda")
            opt.zero_grad()
            pred_acc = nfnet(params).squeeze(-1)
            loss = loss_fn(pred_acc, acc)
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
            if step % 10 == 0:
                theoretical_loss = loss_fn(acc, acc)  # perfect loss
                wandb.log({
                    "train/loss": loss.detach().cpu().item(),
                    "train/rsq": r2_score(acc.cpu().numpy(), pred_acc.detach().cpu().numpy()),
                    "train/theoretical_loss": theoretical_loss.detach().cpu().item(),
                }, step=step)
            step += 1
        # evaluate
        avg_err, avg_loss, rsq, tau, actual, pred = evaluate(nfnet, valloader, loss_fn)
        print(f"Epoch {epoch}, val L1 err: {avg_err:.2f}, val loss: {avg_loss:.2f}, val Rsq: {rsq:.2f}.")
        save_dict = {
            "weights": nfnet.state_dict(),
            "val_l1": avg_err,
            "val_loss": avg_loss,
            "val_rsq": rsq,
            "epoch": epoch,
        }
        if rsq > best_rsq:
            torch.save(save_dict, os.path.join(cfg.output_dir, f"best_rsq.pt"))
            best_rsq = rsq
        if tau > best_tau:
            torch.save(save_dict, os.path.join(cfg.output_dir, f"best_tau.pt"))
            best_tau = tau
        plt.clf()
        plot = plt.scatter(actual, pred)
        plt.xlabel("Actual model accuracy")
        plt.ylabel("Predicted model accuracy")
        wandb.log({
            "val/l1_err": avg_err,
            "val/loss": avg_loss,
            "val/rsq": rsq,
            "val/scatter": wandb.Image(plot),
            "val/kendall_tau": tau,
            "val/best_rsq": best_rsq,
            "val/best_tau": best_tau,
        }, step=step)

    testset = hydra.utils.instantiate(cfg.dset, mode="test")
    if cfg.debug:  # 2 batches for debugging
        testset = torch.utils.data.Subset(testset, range(2 * cfg.batch_size))
    testloader = DataLoader(testset, batch_size=cfg.batch_size, shuffle=False, num_workers=8)
    test_path = os.path.join(cfg.output_dir, f"best_{cfg.test_metric}.pt")
    print(f"Loading best model from {test_path}.")
    nfnet.load_state_dict(torch.load(test_path)["weights"])
    # test
    avg_err, avg_loss, rsq, tau, actual, pred = evaluate(nfnet, testloader, loss_fn)
    print(f"Test L1 err: {avg_err:.2f}, test loss: {avg_loss:.2f}, test Rsq: {rsq:.2f}.")
    plt.clf()
    plot = plt.scatter(actual, pred)
    plt.xlabel("Actual model accuracy")
    plt.ylabel("Predicted model accuracy")
    wandb.log({
        "test/l1_err": avg_err,
        "test/loss": avg_loss,
        "test/rsq": rsq,
        "test/scatter": wandb.Image(plot),
        "test/kendall_tau": tau,
    })

    wandb.finish()