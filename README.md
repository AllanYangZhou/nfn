# Neural Functional Networks (NFNs)
### [[Paper (arXiv)](https://arxiv.org/abs/2302.14040)]

This library provides a PyTorch implementation of layers for building permutation equivariant **neural functional networks** (NFNs). NFNs are equivariant deep learning architectures for processing weight space features, such as the weights or gradients of another neural network. We refer to these layers as **NF-Layers**.

## Installation
Simple installation from PyPI:
```bash
pip install nfn
```
If you want to edit source or run examples, clone the repository locally. Then run the following commands:
```bash
git clone https://github.com/AllanYangZhou/nfn
cd nfn
pip install -e .  # installs in editable mode
```

## Usage

### Example
We provide a very simple example of using this library to process the weights of a small CNN using an NFN at `examples/basic_cnn`. The example only shows how to construct NFNs and feed data in, but does not train anything. To run:
```bash
cd examples/basic_cnn
python check_nfn_inv.py
```
This will check the permutation invariance of the NFN.

### Overview
NF-Layers operate on `WeightSpaceFeatures`. The current NF-Layers are compatible with the weight spaces of simple feedforward MLPs and 2D (image) CNNs. For weight spaces of CNN classifiers we assume there is some global pooling layer (e.g., `nn.AdaptiveAvgPool2d(1)`) between the convolution and FC layers. Supporting 1D or 3D CNNs should be possible but is not currently implemented.

To construct `WeightSpaceFeatures` from the weights of a Pytorch model, we provide the helper function `state_dict_to_tensors`:
```python
models = [...]  # batch of pytorch models
state_dicts = [m.state_dict() for m in models]
wts_and_bs = [state_dict_to_tensors(sd) for sd in state_dicts]
# Collate batch. Can be done automatically by DataLoader.
wts_and_bs = default_collate(wts_and_bs)
wtfeat = WeightSpaceFeatures(*wts_and_bs)

out = nfn(wtfeat)  # NFN can now ingest WeightSpaceFeatures
```
For now, `state_dict_to_tensors` assumes that the `state_dict` is an ordered dictionary with keys in order `[weight1, bias1, ..., weightL, biasL]`. This is the default behavior if the weights are coming from a feedforward network that is an `nn.Sequential` model.
