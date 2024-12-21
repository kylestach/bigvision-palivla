from os import PathLike

import cloudpickle
from einops import rearrange, repeat
import numpy as np
import tensorflow as tf

from big_vision.utils import Registry


class ActionTokenizer:
    def tokenize(self, data, obs=None): ...

    def detokenize(self, tokens, obs=None): ...

    def save(self, path: PathLike):
        with tf.io.gfile.GFile(path / "action_tokenizer.pkl", "wb") as f:
            cloudpickle.dump(self, f)

    @classmethod
    def load(cls, path: PathLike):
        with tf.io.gfile.GFile(path / "action_tokenizer.pkl", "rb") as f:
            return cloudpickle.load(f)


@Registry.register("action_tokenizer.bin")
class BinActionTokenizer(ActionTokenizer):
    def __init__(
        self,
        min_action_value: np.ndarray | float,
        max_action_value: np.ndarray | float,
        action_dim: int | None = None,
        action_vocab_size: int = 1000,
        action_horizon: int = 10,
    ):
        self.min_action_value = min_action_value
        self.max_action_value = max_action_value
        self.action_dim = action_dim
        self.action_vocab_size = action_vocab_size
        self.action_horizon = action_horizon

    @property
    def num_tokens(self):
        return self.action_horizon * self.action_dim

    @property
    def vocab_size(self):
        return self.action_vocab_size

    def tokenize(self, data, obs=None):
        data = (data - self.min_action_value) / (
            self.max_action_value - self.min_action_value
        )
        data = rearrange(data, "... p a -> ... (p a)")
        return np.clip(
            np.round(data * (self.vocab_size - 1)).astype(np.int32),
            0,
            self.vocab_size - 1,
        )

    def detokenize(self, tokens, obs=None, action_dim: int | None = None):
        values = np.where(
            (tokens < 0) | (tokens >= self.vocab_size),
            np.nan,
            tokens / (self.vocab_size - 1)
        )
        data = (
            values * (self.max_action_value - self.min_action_value)
            + self.min_action_value
        )
        data = rearrange(data, "... (p a) -> ... p a", a=action_dim or self.action_dim)
        return data