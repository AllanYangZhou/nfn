import argparse
import os
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import hydra
from tqdm import tqdm

from experiments.data_utils import SirenAndOriginalDataset
from nfn.common import network_spec_from_wsfeat, WeightSpaceFeatures


def main(args):
    cfg = os.path.join(args.rundir, ".hydra/config.yaml")
    cfg = OmegaConf.load(cfg)
    # load data
    print('\n\n\n', cfg.dset.siren_path, cfg.dset.data_path)
    dset = SirenAndOriginalDataset(cfg.dset.siren_path, "randinit_smaller", cfg.dset.data_path)
    loader = DataLoader(dset, batch_size=cfg.bs, shuffle=False, num_workers=8, drop_last=False)

    # load weight features
    spec = network_spec_from_wsfeat(WeightSpaceFeatures(*next(iter(loader))[0]).to("cpu"), set_all_dims=True)
    nfnet = hydra.utils.instantiate(cfg.model, spec, dset_data_type=dset.data_type, vae=False, compile=False).to("cuda")
    nfnet.load_state_dict(torch.load(os.path.join(args.rundir, "best_nfnet.pt")))

    nfnet.eval()
    for param in nfnet.parameters():
        param.requires_grad = False

    # dset.data_type in ["mnist", "fashion", "cifar"]

    # create embedding dataset
    if not os.path.exists(args.output_path):
        print("Computing embeddings...")
        embeddings, labels = [], []
        for wts_and_bs, _, label in tqdm(loader):
            params = WeightSpaceFeatures(*wts_and_bs).to("cuda")
            with torch.no_grad():
                embeddings.append(nfnet.encode(params).cpu())
                labels.append(label)
        embeddings = torch.cat(embeddings)
        labels = torch.cat(labels)
        torch.save({
            "embeddings": embeddings,
            "labels": labels,
            "split_points": cfg.dset.split_points
        }, args.output_path)
        print("Embeddings saved to", args.output_path)
    else:
        print("Embeddings already exists, not doing anything.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rundir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    main(parser.parse_args())