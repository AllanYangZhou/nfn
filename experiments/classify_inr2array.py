import os
import torch
import numpy as np
from torch import nn
from torch.utils import data
from tqdm import tqdm, trange
import wandb
from omegaconf import OmegaConf

from experiments.train_utils import get_linear_warmup_with_cos_decay


class TransformerClassifier(nn.Module):
    def __init__(self, n_classes=10, num_blocks=12):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            512, 4, dim_feedforward=2048, dropout=0.1,
            activation="gelu", layer_norm_eps=1e-05,
            batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_blocks)
        self.proj = nn.Linear(256, 512)
        self.pos = nn.Embedding(64, 512)

        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, n_classes)
        )

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        x = self.proj(x)
        x = x + self.pos(torch.arange(0, x.shape[1]).to(x.device))
        x = self.transformer_encoder(x)
        x = x[:, 0, :]
        x = self.classifier(x)
        return x


def load_data(embedding_path):
    emb_data = torch.load(embedding_path)
    embeddings = emb_data["embeddings"]
    labels = emb_data["labels"]
    split_points = emb_data["split_points"]
    return embeddings, labels, split_points


def make_dataset(embeddings, labels, split_points):
    # construct tensordataset from embeddings and labels
    dataset = data.TensorDataset(embeddings, labels)
    # split into train/val/test sets.
    val_point, test_point = split_points
    trainset, valset = data.Subset(dataset, range(val_point)), data.Subset(dataset, range(val_point, test_point))
    testset = data.Subset(dataset, range(test_point, len(dataset)))
    trainloader = data.DataLoader(trainset, batch_size=128, shuffle=True, drop_last=True)
    valloader = data.DataLoader(valset, batch_size=128, shuffle=False, drop_last=True)
    testloader = data.DataLoader(testset, batch_size=128, shuffle=False, drop_last=True)
    return trainloader, valloader, testloader


@torch.no_grad()
def evaluate(loader, model):
    state = model.training
    model.eval()
    acc = 0
    for x, y in tqdm(loader, position=2, leave=False):
        x, y = x.cuda(), y.cuda()
        logits = model(x)
        acc += (logits.argmax(dim=-1) == y).float().mean().item()
    model.train(state)
    return acc / len(loader)


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def train_and_evaluate(
    clf, opt, sched,
    trainloader,
    valloader,
    testloader,
    n_epochs=10,
    out_dir="",
    mixup_alpha=0,
):
    loss_fn = nn.CrossEntropyLoss()
    best_val_acc = 0
    best_ckpt_path = os.path.join(out_dir, "best.pt")
    outer_pbar = trange(0, n_epochs, position=0)
    global_step = 0
    for epoch_idx in outer_pbar:
        for x, y in tqdm(trainloader, position=1, leave=False):
            opt.zero_grad()
            x, y = x.cuda(), y.cuda()
            x, y_a, y_b, lam = mixup_data(x, y, mixup_alpha)
            logits = clf(x)
            loss = lam * loss_fn(logits, y_a) + (1 - lam) * loss_fn(logits, y_b)
            loss.backward()
            opt.step()
            if sched is not None: sched.step()
            outer_pbar.set_description(f"epoch {epoch_idx} loss: {loss.item():.3f}")
            global_step += 1
            if global_step % 50 == 0:
                train_acc_a = (logits.detach().argmax(dim=-1).cpu().numpy() == y_a.detach().cpu().numpy()).mean().item()
                train_acc_b = (logits.detach().argmax(dim=-1).cpu().numpy() == y_b.detach().cpu().numpy()).mean().item()
                train_acc = lam * train_acc_a + (1 - lam) * train_acc_b
                wandb.log({"train/loss": loss.detach().item(), "train/acc": train_acc}, step=global_step)
        # calculate val accuracy
        val_acc = evaluate(valloader, clf)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(clf.state_dict(), best_ckpt_path)
        outer_pbar.set_description(f"epoch {epoch_idx} val acc: {val_acc:.3f}")
        wandb.log({"val/acc": val_acc, "val/best_acc": best_val_acc}, step=global_step)
    outer_pbar.close()
    final_train_acc = evaluate(trainloader, clf)
    # load best.pt
    clf.load_state_dict(torch.load(best_ckpt_path))
    test_acc = evaluate(testloader, clf)
    print(f"Test accuracy: {test_acc:.3f}. Final train_acc: {final_train_acc:.3f}.")
    wandb.log({"final_train/acc": final_train_acc, "test/acc": test_acc}, step=global_step)


def main(cfg):
    wandb.init(project="classify-latent", reinit=True)
    wandb.config.update(OmegaConf.to_container(cfg, resolve=True))
    # load data
    embeddings, labels, split_points = load_data(cfg.embedding_path)
    # make dataset
    trainloader, valloader, testloader = make_dataset(embeddings, labels, split_points)
    # make model
    clf = TransformerClassifier().cuda()
    opt = torch.optim.AdamW(clf.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    total_steps = len(trainloader) * cfg.n_epochs
    sched = get_linear_warmup_with_cos_decay(opt, total_steps=total_steps, warmup_steps=10_000)
    # train
    train_and_evaluate(clf, opt, sched, trainloader, valloader, testloader, n_epochs=cfg.n_epochs, out_dir=cfg.output_dir, mixup_alpha=cfg.alpha)