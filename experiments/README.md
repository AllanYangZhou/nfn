# Experiments 
This folder contains experiments from [Permutation Equivariant Neural Functionals
](https://arxiv.org/abs/2302.14040) and [Neural Functional Transformers
](https://arxiv.org/abs/2305.13546). Throughout this document we assume that your current working directory is the **root** directory of the repository.

The experiments require additional dependencies. In your virtual env:
```bash
pip install -r requirements.txt
pip install -e experiments/perceiver-pytorch
```
We use Python 3.8.16.

## Predicting generalization for small CNNs
### Getting CNN Zoo data
Download the [CIFAR10](https://storage.cloud.google.com/gresearch/smallcnnzoo-dataset/cifar10.tar.xz) and [SVHN](https://storage.cloud.google.com/gresearch/smallcnnzoo-dataset/svhn_cropped.tar.xz) data  (originally from [Unterthiner et al, 2020](https://github.com/google-research/google-research/tree/master/dnn_predict_accuracy)) into `./experiments/data`, and extract them. Change `data_root_dir` in `main.yaml` if you want to store the data somewhere else.

### Launching training
Some examples of predicting generalization with hypernetworks. Add `+mode=debug` to anything to run a few steps quickly and test the whole training pipeline. 
```sh
python -m experiments.launch_predict_gen dset=[...] nfnet=[...]
```

Options for `dset`:
- `zoo_cifar`: CNN Zoo (CIFAR) dataset
- `zoo_svhn` : CNN Zoo (SVHN) dataset

Options for `nfnet`:
- `hnp_inv`: Invariant NFN using HNP NF-Layers
- `np_inv`: Invariant NFN using NP NF-Layers, with learned IO-encoding.
- `stat`: StatNet, a method from Unterthiner et al, 2020.

## Stylizing SIRENs

We provide datasets of SIREN weights trained on the MNIST, FashionMNIST, and CIFAR image datasets, available from [this link](https://drive.google.com/drive/folders/15CdOTPWHqDcS4SwbIdm100rXkTYZPcC5?usp=sharing). Download these datasets into `./experiments/data` and untar them:
```sh
tar -xvf siren_mnist_wts.tar  # creates a folder siren_mnist_wts/
tar -xvf siren_fashion_wts.tar  # creates a folder siren_fashion_wts/
tar -xvf siren_cifar_wts.tar  # creates a folder siren_cifar_wts/
```

We have two experiment settings:

1. `+setup=mnist`: train an NFN to dilate MNIST digits
1. `+setup=cifar`: train an NFN to increase contrast of CIFAR10 images

And a few architecture options:

* `nfnet=equiv`: HNP-equivariant architecture
* `nfnet=equiv_io`: NP-equivariant architecture
* `nfnet=mlp`: MLP baseline

For example:
```bash
python -m experiments.launch_stylize_siren +setup=mnist nfnet=equiv
```

## Classifying SIRENs
These experiments use the same MNIST, FashionMNIST, and CIFAR-10 INR datasets as in the previous section.

### With INR2Array (NFT)
This produces low-dimensional embeddings of the SIRENs using Neural Functional Transformers (NFTs). Some reasonable hyperparam configs:
```bash
# MNIST
python -m experiments.launch_inr2array dset=mnist model=nft compile=true
# FashionMNIST
python -m experiments.launch_inr2array dset=fashion model=nft compile=true
# CIFAR
python -m experiments.launch_inr2array dset=cifar model=nft_additive compile=true
```
You may also add `model/pool_cls=crossattn_2x2` or `cross_attn_8x8` to vary the size of the spatial latent. `compile` uses Pytorch 2.0's new compile feature which can improve computational performance.

Once the INR2Array model is trained, you want to produce a dataset of embeddings from the SIREN dataset. In this example, `rundir` should be the hydra rundir for the INR2Array model you just trained, which could be something like `./outputs/2023-11-05/23-51-53`.

```bash
python -m experiments.make_latent_dset --rundir [...] --output_path experiments/data/mnist-embeddings.pt
```

Finally, you can train a classifier on the embeddings:

```bash
python -m experiments.launch_classify_latent embedding_path=experiments/data/mnist-embeddings.pt
```
