from os import PathLike
from typing import Any

import cloudpickle
import numpy as np
import tensorflow as tf
from einops import rearrange, EinopsError
from transformers import AutoProcessor

from big_vision.utils import Registry


class ActionTokenizer:
    def tokenize(self, data, obs=None): ...

    def detokenize(self, tokens, obs=None): ...

    def save(self, path: Any):
        with tf.io.gfile.GFile(tf.io.gfile.join(path, "action_tokenizer.pkl"), "wb") as f:
            cloudpickle.dump(self, f)

    @classmethod
    def load(cls, path: PathLike):
        with tf.io.gfile.GFile(tf.io.gfile.join(path, "action_tokenizer.pkl"), "rb") as f:
            return cloudpickle.load(f)


@Registry.register("action_tokenizer.bin")
class BinActionTokenizer(ActionTokenizer):
    def __init__(
        self,
        min_action_value: np.ndarray | float,
        max_action_value: np.ndarray | float,
        action_vocab_size: int = 1000,
        action_horizon: int = 1,
    ):
        self.min_action_value = min_action_value
        self.max_action_value = max_action_value
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

    def detokenize(self, tokens, *, obs=None, action_dim: int):
        values = np.where(
            (tokens < 0) | (tokens >= self.vocab_size),
            np.nan,
            tokens / (self.vocab_size - 1),
        )
        data = (
            values * (self.max_action_value - self.min_action_value)
            + self.min_action_value
        )
        data = data[..., :action_dim]
        try:
            data = rearrange(data, "... (p a) -> ... p a", a=action_dim)
        except EinopsError:
            raise ValueError(f"Could not detokenize data with shape {data.shape} into {action_dim} dimensions")
        return data

@Registry.register("action_tokenizer.fast")
class FASTActionTokenizer(ActionTokenizer):
    def __init__(
        self, 
        min_action_value: np.ndarray | float,
        max_action_value: np.ndarray | float,
        action_vocab_size: int = 1024,
        # add chunking ! 
    ):
        self.tokenizer = AutoProcessor.from_pretrained(
            "physical-intelligence/fast", trust_remote_code=True
        )
        self.min_action_value = min_action_value
        self.max_action_value = max_action_value
        self.action_vocab_size = action_vocab_size

    @property
    def vocab_size(self):
        return self.action_vocab_size

    def tokenize(self, data, obs=None):
        data = -1 + 2 * (data - self.min_action_value) / (
            self.max_action_value - self.min_action_value
        ) # normalize to [-1, 1]

        return self.tokenizer(data)

    def detokenize(self, tokens, *, obs=None, action_dim: int):
        # if there are any invalid tokens (i.e. >1024 or <0), the action is deemed invalid
        if np.any((tokens < 0) | (tokens >= self.vocab_size)):
            invalid_action = np.empty(shape=(tokens.shape[0], action_dim))
            invalid_action.fill(np.nan)
            return invalid_action

        # the issue is that this token sequence might not always correspond to 14 tokens
        # so this might still error... 
        unnormalized_actions = self.tokenizer.decode([tokens], action_dim=14)  
        data = self.min_action_value + (unnormalized_actions + 1) * 0.5 * (
            self.max_action_value - self.min_action_value
        ) # unnormalize from [-1,1] to original range

        data = data.squeeze(0) # unbatch

        return data