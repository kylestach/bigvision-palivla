from big_vision.utils import Registry
from palivla.model_components import ModelComponents


@Registry.register("load.paligemma_weights")
def load_paligemma_weights(model: ModelComponents, *, path: str):
    pass
