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

    def save(self, path: PathLike):
        with tf.io.gfile.GFile(path / "sequence_builder.pkl", "wb") as f:
            cloudpickle.dump(self, f)

    @classmethod
    def load(cls, path: PathLike):
        with tf.io.gfile.GFile(path / "sequence_builder.pkl", "rb") as f:
            return cloudpickle.load(f)

    def prepare_prompt(self, language_instruction):
        return "<bos>" + str(language_instruction)

    def prepare_gen(self, action_tokens):
        return "".join([f"<act{i}>" for i in action_tokens]) + "<eos>"

    def build_sequence(
        self,
        batch,
        language_tokenizer: AutoTokenizer,
        action_tokenizer: ActionTokenizer,
        boa_is_prompt: bool = False,
    ):
        boa_id = language_tokenizer.encode("<begin_of_action>")[0]

        boa_prompt = "<begin_of_action>" if boa_is_prompt else ""
        boa_gen = "" if boa_is_prompt else "<begin_of_action>"
        prompt = [
            self.prepare_prompt(instruction) + boa_prompt
            for instruction in batch["task"]["language_instruction"]
        ]
        action_tokens = [
            boa_gen + self.prepare_gen(t)
            for t in action_tokenizer.tokenize(batch["action"][..., -1, :, :])
        ]

        prompt_tokens = language_tokenizer.batch_encode_plus(prompt)["input_ids"]
        action_tokens = language_tokenizer.batch_encode_plus(action_tokens)["input_ids"]

        def _pad(data, pad_length, pad_value=0):
            num_pad_tokens = max(0, pad_length - len(data))
            return np.pad(
                data, (0, num_pad_tokens), mode="constant", constant_values=pad_value
            )[:pad_length]

        batch_size = len(prompt_tokens)

        return {
            "prompt": {
                "tokens": np.stack(
                    [_pad(tok, self.prompt_pad_length) for tok in prompt_tokens]
                ),
                "mask": np.stack(
                    [
                        _pad(np.ones_like(tok, dtype=bool), self.prompt_pad_length)
                        for tok in prompt_tokens
                    ]
                ),
                "mask_ar": np.stack(
                    [
                        _pad(np.equal(tok, boa_id), self.prompt_pad_length)
                        for tok in prompt_tokens
                    ]
                ),
                "mask_loss": np.zeros((batch_size, self.prompt_pad_length), dtype=bool),
            },
            "gen": {
                "tokens": np.stack(
                    [_pad(tok, self.gen_pad_length) for tok in action_tokens]
                ),
                "mask": np.stack(
                    [
                        _pad(np.ones_like(tok, dtype=bool), self.gen_pad_length)
                        for tok in action_tokens
                    ]
                ),
                "mask_ar": np.ones((batch_size, self.gen_pad_length), dtype=bool),
                "mask_loss": np.stack(
                    [
                        _pad(np.ones(len(tok), dtype=bool), self.gen_pad_length)
                        for tok in action_tokens
                    ]
                ),
            },
        }

    def get_actions(
        self,
        tokens: np.ndarray,
        language_tokenizer: AutoTokenizer,
        action_tokenizer: ActionTokenizer,
        *,
        boa_is_prompt: bool = False,
        action_dim: int | None,
        boa_id: int | None = None,
        eos_id: int | None = None,
        act0_id: int | None = None,
    ):
        boa_id = boa_id or language_tokenizer.encode("<begin_of_action>")[0]
        eos_id = eos_id or language_tokenizer.encode("<eos>")[0]
        act0_id = act0_id or language_tokenizer.encode("<act0>")[0]

        # Find the beginning of the action
        if boa_is_prompt:
            start_idx = 0
        else:
            try:
                start_idx = np.where(tokens == boa_id)[0][0] + 1
            except IndexError:
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
        except:
            return None

    def batch_get_actions(
        self,
        tokens,
        language_tokenizer: AutoTokenizer,
        action_tokenizer: ActionTokenizer,
        *,
        boa_is_prompt: bool = False,
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
                boa_is_prompt=boa_is_prompt,
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
                action_dim = action.shape[1]
                break

        actions_mask = np.array([action is not None for action in actions])
        actions = np.stack(
            [
                (
                    np.pad(action, ((0, action_horizon - action.shape[0]), (0, 0)), constant_values=np.nan)
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
