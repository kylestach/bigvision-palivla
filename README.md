# PaliVLA
This is a framework for training multimodal vision-language-action (VLA) model for robotics in JAX. It primarily supports PaliGemma for now, though more base models will be added in the future.

## Installation
We develop with `uv`, but other environment managers should work fine. To install the dependencies, run:
```bash
uv venv
uv sync
```

## Training
To train a model, run:
```bash
python -m palivla/train.py --config_file palivla/configs/bridge_config.py
```

This repository is (for now) a fork of [`big_vision`](https://github.com/google-research/big_vision).

## Citation
If you use PaliVLA in your own project, please cite this repository:
```bibtex
@misc{palivla,
  author       = {Kyle Stachowicz},
  title        = {PaliVLA},
  year         = {2024},
  url          = {https://github.com/kylestach/bigvision-palivla},
  note         = {GitHub repository}
}
```
