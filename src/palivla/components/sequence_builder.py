from dataclasses import dataclass
from os import PathLike
import cloudpickle
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
