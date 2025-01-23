from dataclasses import dataclass
from os import PathLike

import cloudpickle
import einops
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer

from big_vision.utils import Registry
from palivla.components.action_tokenizer import ActionTokenizer
from palivla.components.sequence_builder import SequenceBuilder


@Registry.register("sequence_builder.cot")
@dataclass
class CoTSequenceBuilder(SequenceBuilder):
    def prepare_cot(self, reasonings):
        if isinstance(reasonings, bytes):
            reasonings = reasonings.decode("utf-8")

        assert isinstance(reasonings, str)

        return str(reasonings)

    def prepare_gen(self, reasonings, action_tokens):
        if isinstance(reasonings, bytes):
            reasonings = reasonings.decode("utf-8")

        assert isinstance(reasonings, str)

        return (
            str(reasonings) + "<begin_of_action>" + super().prepare_gen(action_tokens)
        )

    def build_sequence(
        self,
        batch,
        language_tokenizer: AutoTokenizer,
        action_tokenizer: ActionTokenizer,
        begin_is_prompt: bool = False,
        include_action_tokens: bool = True,
    ):
        boa_id = language_tokenizer.encode("<begin_of_action>")[0]
        bor_id = language_tokenizer.encode("<begin_of_reasoning>")[0]

        boa_prompt = "<begin_of_reasoning>" if begin_is_prompt else ""
        boa_gen = "" if begin_is_prompt else "<begin_of_reasoning>"
        prompt = [
            self.prepare_prompt(instruction) + boa_prompt
            for instruction in batch["task"]["language_instruction"]
        ]

        prompt_tokens = language_tokenizer.batch_encode_plus(prompt)["input_ids"]
        
        if include_action_tokens:
            action = batch["action"]
            if action.ndim == 4:
                action = action[..., -1, :, :]

            gen_tokens = [
                boa_gen + self.prepare_gen(reasonings, t)
                for reasonings, t in zip(
                    batch["reasonings"], action_tokenizer.tokenize(action)
                )
            ]
            gen_tokens = language_tokenizer.batch_encode_plus(gen_tokens)["input_ids"]
        else:
            gen_tokens = [[] for _ in range(len(prompt_tokens))]

        prompt_is_ar = [[tok == bor_id for tok in prompt] for prompt in prompt_tokens]
        return {
            "prompt": self.pad_and_format_token_group(
                prompt_tokens,
                self.prompt_pad_length,
                is_gen=prompt_is_ar,
                is_loss=False,
            ),
            "gen": self.pad_and_format_token_group(
                gen_tokens, self.gen_pad_length, is_gen=True, is_loss=True
            ),
        }

    def get_chain_of_thought(
        self,
        tokens: np.ndarray,
        language_tokenizer: AutoTokenizer,
        *,
        bor_id: int | None = None,
    ):
        if bor_id is None:
            bor_id = language_tokenizer.encode("<begin_of_reasoning>")[0]

        # Find the beginning of the action (end of CoT)
        try:
            end_idx = np.where(tokens == bor_id)[0][0]
            tokens = tokens[:end_idx]
        except IndexError:
            tokens = tokens

        return language_tokenizer.decode(tokens)
