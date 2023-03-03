# Experiments (WIP)
This folder will contain experiments corresponding to [the paper](https://arxiv.org/abs/2302.14040). Throughout this document we assume that your current working directory is the root directory of the repository.

The experiments require additional dependencies. In your virtual env:
```bash
pip install -r requirements.txt
```
We use Python 3.7.12.

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
