# PaliVLA
This is a framework for training multimodal vision-language-action (VLA) model for robotics in JAX. It primarily supports PaliGemma for now, though more base models will be added in the future.

## Installation
We develop with `uv`, but other environment managers should work fine. To install the dependencies, run:
```bash
uv init --python=python3.10
uv sync
```

We require some extra packages, namely [`dlimp`](https://github.com/kvablack/dlimp) and [`octo`](https://github.com/octo-models/octo) for dataloading. To install them, clone them locally. You will have to install `dlimp` without its dependencies, as its specified tensorflow version has some mutual incompatibilities with other packages we use.

## Training
To train a model, run:
```bash
python -m palivla/train.py --config_file palivla/configs/bridge_config.py
```

This repository is (for now) a fork of [`big_vision`](https://github.com/google-research/big_vision).
