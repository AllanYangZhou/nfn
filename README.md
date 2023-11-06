# Neural Functional Networks (NFNs)
### Papers: [Permutation Equivariant Neural Functionals](https://arxiv.org/abs/2302.14040) and [Neural Functional Transformers](https://arxiv.org/abs/2305.13546)

![Diagram of NFN](/imgs/diagram.png)

This is a library of PyTorch layers for building permutation equivariant **neural functional networks** (NFNs). NFNs are equivariant deep learning architectures for processing weight space features, such as the weights or gradients of another neural network. We refer to the layers of an NFN as **NF-Layers**.

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

### Loading weights as input
NF-Layers operate on `WeightSpaceFeatures`. The current NF-Layers are compatible with the weight spaces of simple feedforward MLPs and 2D (image) CNNs. For weight spaces of CNN classifiers we assume there is some global pooling layer (e.g., `nn.AdaptiveAvgPool2d(1)`) between the convolution and FC layers. Supporting 1D or 3D CNNs should be possible but is not currently implemented.

To construct `WeightSpaceFeatures` from the weights of a Pytorch model, we provide the helper function `state_dict_to_tensors`:
```python
from nfn.common import state_dict_to_tensors

models = [...]  # batch of pytorch models
state_dicts = [m.state_dict() for m in models]
wts_and_bs = [state_dict_to_tensors(sd) for sd in state_dicts]
# Collate batch. Can be done automatically by DataLoader.
wts_and_bs = default_collate(wts_and_bs)
wsfeat = WeightSpaceFeatures(*wts_and_bs)

out = nfn(wsfeat)  # NFN can now ingest WeightSpaceFeatures
```
For now, `state_dict_to_tensors` assumes that the `state_dict` is an ordered dictionary with keys in order `[weight1, bias1, ..., weightL, biasL]`. This is the default behavior if the `state_dict` is coming from a feedforward network that is an `nn.Sequential` model.

### Building NFNs
The NF-Layers are found in `nfn.layers`. The main data you need to build an NFN is a `network_spec`, which specifies the structure of the weight space you plan to process. If you already have a `WeightSpaceFeature` object as above, you can use `nfn.common.network_spec_from_wsfeat`.

```python
from torch import nn
from nfn import layers
from nfn.common import network_spec_from_wsfeat

network_spec = network_spec_from_wsfeat(wsfeat)
nfn_channels = 32

# io_embed: encode the input and output dimensions of the weight space feature
nfn = nn.Sequential(
    layers.NPLinear(network_spec, 1, nfn_channels, io_embed=True),
    layers.TupleOp(nn.ReLU()),
    layers.NPLinear(network_spec, nfn_channels, nfn_channels, io_embed=True),
    layers.TupleOp(nn.ReLU()),
    layers.HNPPool(network_spec),  # pooling layer, for invariance
    nn.Flatten(start_dim=-2),
    nn.Linear(nfn_channels * layers.HNPPool.get_num_outs(network_spec), 1)
)
```
