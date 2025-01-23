from dataclasses import dataclass
from os import PathLike

import cloudpickle
import einops
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer

from big_vision.utils import Registry
from palivla.components.action_tokenizer import ActionTokenizer


@Registry.register("sequence_builder.default")
@dataclass
class SequenceBuilder:
    prompt_pad_length: int
    gen_pad_length: int

    @property
    def max_decode_length(self):
        return self.gen_pad_length

    def save(self, path: PathLike):
        with tf.io.gfile.GFile(tf.io.gfile.join(path, "sequence_builder.pkl"), "wb") as f:
            cloudpickle.dump(self, f)

    @classmethod
    def load(cls, path: PathLike):
        with tf.io.gfile.GFile(tf.io.gfile.join(path, "sequence_builder.pkl"), "rb") as f:
            return cloudpickle.load(f)

    def prepare_prompt(self, language_instruction):
        if isinstance(language_instruction, bytes):
            language_instruction = language_instruction.decode()
        
        assert isinstance(language_instruction, str)

        return "<bos>" + language_instruction

    def prepare_gen(self, action_tokens):
        return "".join([f"<act{i}>" for i in action_tokens]) + "<eos>"

    def build_sequence(
        self,
        batch,
        language_tokenizer: AutoTokenizer,
        action_tokenizer: ActionTokenizer,
        begin_is_prompt: bool = False,
        include_action_tokens: bool = True,
    ):
        boa_id = language_tokenizer.encode("<begin_of_action>")[0]

        boa_prompt = "<begin_of_action>" if begin_is_prompt else ""
        boa_gen = "" if begin_is_prompt else "<begin_of_action>"
        prompt = [
            self.prepare_prompt(instruction) + boa_prompt
            for instruction in batch["task"]["language_instruction"]
        ]

        prompt_tokens = language_tokenizer.batch_encode_plus(prompt)["input_ids"]
        
        if include_action_tokens:
            action = batch["action"]
            if action.ndim == 4:
                action = action[..., -1, :, :]

            action_tokens = [
                boa_gen + self.prepare_gen(t)
                for t in action_tokenizer.tokenize(action)
            ]
            action_tokens = language_tokenizer.batch_encode_plus(action_tokens)[
                "input_ids"
            ]
        else:
            action_tokens = [[] for _ in range(len(prompt_tokens))]

        prompt_is_ar = [[tok == boa_id for tok in prompt] for prompt in prompt_tokens]
        return {
            "prompt": self.pad_and_format_token_group(prompt_tokens, self.prompt_pad_length, is_gen=prompt_is_ar, is_loss=False),
            "gen": self.pad_and_format_token_group(action_tokens, self.gen_pad_length, is_gen=True, is_loss=True),
        }
   
    @staticmethod
    def pad_and_format_token_group(tokens, pad_length, is_gen: bool | np.ndarray, is_loss: bool | np.ndarray):
        def _pad(data, pad_length, pad_value=0, data_len=None):
            if isinstance(data, bool):
                data = np.full(data_len, data)

            num_pad_tokens = max(0, pad_length - len(data))
            return np.pad(
                data, (0, num_pad_tokens), mode="constant", constant_values=pad_value
            )[:pad_length]

        if isinstance(is_gen, bool):
            is_gen = [is_gen for _ in tokens]
        if isinstance(is_loss, bool):
            is_loss = [is_loss for _ in tokens]

        return {
            "tokens": np.stack([_pad(tok, pad_length) for tok in tokens]),
            "mask": np.stack([_pad(np.ones_like(tok, dtype=bool), pad_length) for tok in tokens]),
            "mask_ar": np.stack([_pad(gen, pad_length, data_len=len(tok)) for tok, gen in zip(tokens, is_gen)]),
            "mask_loss": np.stack([_pad(loss, pad_length, data_len=len(tok)) for tok, loss in zip(tokens, is_loss)]),
        }

    def get_actions(
        self,
        tokens: np.ndarray,
        language_tokenizer: AutoTokenizer,
        action_tokenizer: ActionTokenizer,
        *,
        begin_is_prompt: bool = False,
        action_dim: int | None,
        boa_id: int | None = None,
        eos_id: int | None = None,
        act0_id: int | None = None,
    ):
        boa_id = boa_id or language_tokenizer.encode("<begin_of_action>")[0]
        eos_id = eos_id or language_tokenizer.encode("<eos>")[0]
        act0_id = act0_id or language_tokenizer.encode("<act0>")[0]

        # Find the beginning of the action
        try:
            start_idx = np.where(tokens == boa_id)[0][0] + 1
        except IndexError:
            if begin_is_prompt:
                start_idx = 0
            else:
                return None

        # Find the end of the action
        try:
            end_idx = np.where(tokens == eos_id)[0][0]
        except IndexError:
            return None

        # Get the action
        action = tokens[start_idx:end_idx] - act0_id
        try:
            return action_tokenizer.detokenize(action, action_dim=action_dim)
        except ValueError:
            return None

    def batch_get_actions(
        self,
        tokens,
        language_tokenizer: AutoTokenizer,
        action_tokenizer: ActionTokenizer,
        *,
        begin_is_prompt: bool = False,
        action_dim: int,
    ):
        boa_id = language_tokenizer.encode("<begin_of_action>")[0]
        eos_id = language_tokenizer.encode("<eos>")[0]
        act0_id = language_tokenizer.encode("<act0>")[0]

        actions = [
            self.get_actions(
                tokens[i],
                language_tokenizer,
                action_tokenizer,
                begin_is_prompt=begin_is_prompt,
                boa_id=boa_id,
                eos_id=eos_id,
                act0_id=act0_id,
                action_dim=action_dim,
            )
            for i in range(len(tokens))
        ]

        # Get the shape of a valid action
        action_horizon = 0
        for action in actions:
            if action is not None:
                action_horizon = max(action_horizon, action.shape[0])
                if action_dim is None:
                    action_dim = action.shape[1]
                assert action_dim == action.shape[1]

        actions_mask = np.array([action is not None for action in actions])
        actions = np.stack(
            [
                (
                    np.pad(
                        action,
                        ((0, action_horizon - action.shape[0]), (0, 0)),
                        constant_values=np.nan,
                    )
                    if action is not None
                    else np.zeros((action_horizon, action_dim))
                )
                for action in actions
            ]
        )
        actions_mask = einops.repeat(
            actions_mask, "b -> b p a", p=action_horizon, a=action_dim
        ) & ~np.isnan(actions)

        return actions, actions_mask
