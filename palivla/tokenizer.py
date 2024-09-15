from flax import struct
from tensorflow_text import SentencepieceTokenizer
import tensorflow as tf
from ml_collections import ConfigDict
import jax
import numpy as np
import jax.numpy as jnp


@struct.dataclass
class Tokenizer:
    @struct.dataclass
    class TokenizerConfig:
        action_vocab_size: int = struct.field(pytree_node=False)
        action_vocab_offset: int = struct.field(pytree_node=False)
        vocab_size: int = struct.field(pytree_node=False)
        num_action_tokens: int = struct.field(pytree_node=False)
        bos_token: int = struct.field(pytree_node=True)
        eos_token: int = struct.field(pytree_node=True)
        pad_token: int = struct.field(pytree_node=True)
        begin_of_action_token: int = struct.field(pytree_node=True)
        max_pad_length: int = struct.field(pytree_node=False)
        min_action_value: float = struct.field(pytree_node=True)
        max_action_value: float = struct.field(pytree_node=True)
        prompt_autoregressive: bool = struct.field(pytree_node=True)

        def bin_tokenize(self, data):
            # Assume normalization and clipping to [-1, 1]
            data = tf.clip_by_value(data, self.min_action_value, self.max_action_value)
            data = (data - self.min_action_value) / (
                self.max_action_value - self.min_action_value
            )
            return (
                tf.clip_by_value(
                    tf.cast(data * self.action_vocab_size, tf.int32),
                    0,
                    self.action_vocab_size - 1,
                )
                + self.action_vocab_offset
            )

        def bin_detokenize(self, tokens):
            _np = jax.numpy if isinstance(tokens, jax.Array) else np
            values = (tokens - self.action_vocab_offset) / self.action_vocab_size
            values = _np.where((values < 0) | (values > 1), _np.nan, values)
            data = (
                values * (self.max_action_value - self.min_action_value)
                + self.min_action_value
            )
            return data

    config: TokenizerConfig = struct.field(pytree_node=True)
    language_tokenizer: SentencepieceTokenizer = struct.field(pytree_node=False)
    token_structure: dict = struct.field(pytree_node=False)

    @classmethod
    def from_tokenizer(cls, tokenizer: SentencepieceTokenizer, prompt_autoregressive: bool = False):
        bos_token = tokenizer.string_to_id("<bos>").numpy().item()
        eos_token = tokenizer.string_to_id("<eos>").numpy().item()
        pad_token = tokenizer.string_to_id("<pad>").numpy().item()
        begin_of_action_token = tokenizer.string_to_id("\n").numpy().item()
        max_pad_length = 60

        return cls(
            config=cls.TokenizerConfig(
                action_vocab_size=256,
                action_vocab_offset=256000,
                num_action_tokens=7,
                bos_token=bos_token,
                eos_token=eos_token,
                pad_token=pad_token,
                begin_of_action_token=begin_of_action_token,
                max_pad_length=max_pad_length,
                min_action_value=-2,
                max_action_value=2,
                vocab_size=tokenizer.vocab_size().numpy().item(),
                prompt_autoregressive=prompt_autoregressive,
            ),
            language_tokenizer=tokenizer,
            token_structure={
                "prefix": [
                    [bos_token],
                    "prompt",
                    [begin_of_action_token],
                ],
                "causal": [
                    "action",
                ],
                "pad": [[pad_token] * max_pad_length],
            },
        )

    def compose_token_structure(self, tokens, include_keys=["prefix", "causal", "pad"]):
        def _extract_tokens(ids_or_str):
            if isinstance(ids_or_str, str):
                return tokens[ids_or_str]
            else:
                return tf.constant(ids_or_str, dtype=tf.int32)

        tokens_by_name = {
            k: (
                tf.concat([_extract_tokens(token) for token in v], axis=0)
                if k in include_keys
                else tf.zeros((0,), dtype=tf.int32)
            )
            for k, v in self.token_structure.items()
        }

        tokens = tf.concat(
            [tokens_by_name["prefix"], tokens_by_name["causal"], tokens_by_name["pad"]],
            axis=0,
        )[: self.config.max_pad_length]
        include_prefix_mask = tf.ones_like(tokens_by_name["prefix"], dtype=tf.bool) if self.config.prompt_autoregressive else tf.zeros_like(tokens_by_name["prefix"], dtype=tf.bool)
        mask_ar = tf.concat(
            [
                include_prefix_mask,
                tf.ones_like(tokens_by_name["causal"], dtype=tf.bool),
                tf.ones_like(tokens_by_name["pad"], dtype=tf.bool),
            ],
            axis=0,
        )[: self.config.max_pad_length]
        mask_loss = tf.concat(
            [
                include_prefix_mask,
                tf.ones_like(tokens_by_name["causal"], dtype=tf.bool),
                tf.zeros_like(tokens_by_name["pad"], dtype=tf.bool),
            ],
            axis=0,
        )[: self.config.max_pad_length]

        return tokens, mask_ar, mask_loss

    def tokenize_language_instruction(self, data):
        instruction = data["task"]["language_instruction"]
        instruction = tf.strings.lower(instruction)
        instruction = tf.strings.regex_replace(instruction, "[.?!]", "")
        instruction = tf.strings.regex_replace(instruction, "\n", " ")
        instruction = tf.strings.strip(instruction)
        instruction = tf.strings.join([tf.constant("act "), instruction])

        data["language_instruction_tokens"] = self.language_tokenizer.tokenize(
            instruction
        )

        return data

    def bin_tokenize(self, data):
        return self.config.bin_tokenize(data)

    def bin_detokenize(self, tokens):
        return self.config.bin_detokenize(tokens)

    def prepare_tokens_for_training(self, data):
        tokens = {
            "prompt": data["language_instruction_tokens"][
                : self.config.max_pad_length - 10
            ],
            "action": self.bin_tokenize(tf.squeeze(data["action"], axis=(0, 1))),
        }

        tokens, mask_ar, mask_loss = self.compose_token_structure(tokens)

        data["tokens"] = tokens
        data["mask_ar"] = mask_ar
        data["mask_loss"] = mask_loss
        data["mask_input"] = tokens != self.config.pad_token

        del data["language_instruction_tokens"]

        return data

    def prepare_tokens_for_generation(self, data):
        tokens = {
            "prompt": data["language_instruction_tokens"][
                : self.config.max_pad_length - 10
            ],
        }

        tokens, mask_ar, mask_loss = self.compose_token_structure(
            tokens, include_keys={"prefix", "pad"}
        )

        del data["language_instruction_tokens"]

        data["tokens"] = tokens
        data["mask_ar"] = mask_ar
        data["mask_input"] = tokens != self.config.pad_token

        return data

    def extract_action(self, data):
        action_start = (
            jnp.argmax(data["tokens"] == self.config.begin_of_action_token, axis=-1) + 1
        )
        action_data = data["tokens"][
            action_start : action_start + self.config.num_action_tokens
        ]
        action_data = self.bin_detokenize(action_data)
        return action_data
