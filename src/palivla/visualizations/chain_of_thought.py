import io
from typing import Any
from PIL import Image
import wandb

import matplotlib.pyplot as plt
import numpy as np
import prettytable

import jax
from big_vision.utils import Registry
from palivla.model_components import ModelComponents


@Registry.register("viz.chain_of_thought")
def chain_of_thought(model: ModelComponents, trajectory: Any):
    pass
