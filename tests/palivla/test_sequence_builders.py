import numpy as np
import pytest
from transformers import AutoTokenizer

from palivla.components.sequence_builder import SequenceBuilder
from palivla.components.cot_sequence_builder import CoTSequenceBuilder
from palivla.components.action_tokenizer import BinActionTokenizer


@pytest.fixture
def language_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("google/paligemma2-3b-pt-224")
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token to be the EOS token
    tokenizer.padding_side = 'right'  # Pad on the right side
    tokenizer.padding = True  # Enable padding
    tokenizer.add_tokens(["<begin_of_action>", "<begin_of_reasoning>"] + [
        f"<act{i}>" for i in range(10)
    ])
    return tokenizer


@pytest.fixture
def action_tokenizer():
    return BinActionTokenizer(
        min_action_value=0.0,
        max_action_value=1.0,
        action_vocab_size=10,
        action_horizon=1
    )


@pytest.fixture
def sequence_builder():
    return SequenceBuilder(prompt_pad_length=50, gen_pad_length=30)


@pytest.fixture
def cot_sequence_builder():
    return CoTSequenceBuilder(prompt_pad_length=50, gen_pad_length=30)


class TestSequenceBuilder:
    def test_prepare_prompt(self, sequence_builder):
        instruction = "Pick up the red block"
        expected = "<bos>Pick up the red block"
        assert sequence_builder.prepare_prompt(instruction) == expected

        # Test with bytes input
        instruction_bytes = b"Pick up the red block"
        assert sequence_builder.prepare_prompt(instruction_bytes) == expected

    def test_prepare_gen(self, sequence_builder):
        action_tokens = [1, 2, 3]
        expected = "<act1><act2><act3><eos>"
        assert sequence_builder.prepare_gen(action_tokens) == expected

    def test_build_sequence(self, sequence_builder, language_tokenizer, action_tokenizer):
        batch = {
            "task": {
                "language_instruction": ["Pick up the red block"]
            },
            "action": np.random.rand(1, 1, 10)  # Random action tensor
        }
        
        result = sequence_builder.build_sequence(
            batch, language_tokenizer, action_tokenizer
        )
        
        assert isinstance(result, dict)
        assert "prompt" in result
        assert "gen" in result
        
        # Check shapes
        assert result["prompt"]["tokens"].shape == (1, 50)
        assert result["gen"]["tokens"].shape == (1, 30)
        
        # Check masks
        assert result["prompt"]["mask"].shape == (1, 50)
        assert result["prompt"]["mask_ar"].shape == (1, 50)
        assert result["prompt"]["mask_loss"].shape == (1, 50)
        
        assert result["gen"]["mask"].shape == (1, 30)
        assert result["gen"]["mask_ar"].shape == (1, 30)
        assert result["gen"]["mask_loss"].shape == (1, 30)

    def test_build_sequence_begin_is_prompt(self, sequence_builder, language_tokenizer, action_tokenizer):
        batch = {
            "task": {
                "language_instruction": ["Pick up the red block"]
            },
            "action": np.random.rand(1, 1, 10)  # Random action tensor
        }
        
        result = sequence_builder.build_sequence(
            batch, language_tokenizer, action_tokenizer, begin_is_prompt=True
        )
        
        # Get the token ID for <begin_of_action>
        boa_id = language_tokenizer.encode("<begin_of_action>")[0]
        
        # Find where the <begin_of_action> token is in the prompt
        prompt_tokens = result["prompt"]["tokens"][0]
        boa_positions = np.where(prompt_tokens == boa_id)[0]
        
        # Verify <begin_of_action> is in prompt
        assert len(boa_positions) == 1, "Should have exactly one <begin_of_action> token in prompt"
        boa_pos = boa_positions[0]
        
        # Verify it's marked as causal in mask_ar
        assert result["prompt"]["mask_ar"][0, boa_pos], "<begin_of_action> should be marked as causal"
        
        # Verify it's not in generation sequence
        gen_tokens = result["gen"]["tokens"][0]
        assert boa_id not in gen_tokens, "<begin_of_action> should not be in generation sequence"

    def test_pad_tokens_have_zero_mask(self, sequence_builder, language_tokenizer, action_tokenizer):
        batch = {
            "task": {
                "language_instruction": ["hi"]  # Very short to ensure padding
            },
            "action": np.random.rand(1, 1, 10)
        }
        
        result = sequence_builder.build_sequence(
            batch, language_tokenizer, action_tokenizer
        )
        
        # Get pad token ID
        pad_id = language_tokenizer.encode("<pad>")[0]
        
        # Find pad tokens in prompt
        prompt_tokens = result["prompt"]["tokens"][0]
        pad_positions = np.where(prompt_tokens == pad_id)[0]
        
        # Verify all masks are zero for pad tokens in prompt
        assert len(pad_positions) > 0, "Test requires padding in prompt"
        assert not result["prompt"]["mask"][0][pad_positions].any(), "Mask should be zero for pad tokens"
        assert not result["prompt"]["mask_ar"][0][pad_positions].any(), "mask_ar should be zero for pad tokens"
        assert not result["prompt"]["mask_loss"][0][pad_positions].any(), "mask_loss should be zero for pad tokens"
        
        # Find pad tokens in generation
        gen_tokens = result["gen"]["tokens"][0]
        pad_positions = np.where(gen_tokens == pad_id)[0]
        
        # Verify all masks are zero for pad tokens in generation
        assert len(pad_positions) > 0, "Test requires padding in generation"
        assert not result["gen"]["mask"][0][pad_positions].any(), "Mask should be zero for pad tokens"
        assert not result["gen"]["mask_ar"][0][pad_positions].any(), "mask_ar should be zero for pad tokens"
        assert not result["gen"]["mask_loss"][0][pad_positions].any(), "mask_loss should be zero for pad tokens"


class TestCoTSequenceBuilder:
    def test_prepare_cot(self, cot_sequence_builder):
        reasoning = "First, I need to locate the red block"
        assert cot_sequence_builder.prepare_cot(reasoning) == reasoning

        # Test with bytes input
        reasoning_bytes = b"First, I need to locate the red block"
        assert cot_sequence_builder.prepare_cot(reasoning_bytes) == reasoning

    def test_prepare_gen(self, cot_sequence_builder):
        reasoning = "First, I need to locate the red block"
        action_tokens = [1, 2, 3]
        expected = f"{reasoning}<begin_of_action><act1><act2><act3><eos>"
        assert cot_sequence_builder.prepare_gen(reasoning, action_tokens) == expected

    def test_build_sequence(self, cot_sequence_builder, language_tokenizer, action_tokenizer):
        batch = {
            "task": {
                "language_instruction": ["Pick up the red block"]
            },
            "action": np.random.rand(1, 1, 10),  # Random action tensor
            "reasonings": ["First, I need to locate the red block"]
        }
        
        result = cot_sequence_builder.build_sequence(
            batch, language_tokenizer, action_tokenizer
        )
        
        assert isinstance(result, dict)
        assert "prompt" in result
        assert "gen" in result
        
        # Check shapes
        assert result["prompt"]["tokens"].shape == (1, 50)
        assert result["gen"]["tokens"].shape == (1, 30)
        
        # Check masks
        assert result["prompt"]["mask"].shape == (1, 50)
        assert result["prompt"]["mask_ar"].shape == (1, 50)
        assert result["prompt"]["mask_loss"].shape == (1, 50)
        
        assert result["gen"]["mask"].shape == (1, 30)
        assert result["gen"]["mask_ar"].shape == (1, 30)
        assert result["gen"]["mask_loss"].shape == (1, 30)

    def test_get_chain_of_thought(self, cot_sequence_builder, language_tokenizer):
        # Create a sequence with a reasoning part
        test_text = "This is the reasoning<begin_of_reasoning>more text"
        tokens = language_tokenizer.encode(test_text)
        
        result = cot_sequence_builder.get_chain_of_thought(
            np.array(tokens), language_tokenizer
        )
        
        # Should only contain the text before <begin_of_reasoning>
        assert "This is the reasoning" in result
        assert "more text" not in result

    def test_build_sequence_begin_is_prompt(self, cot_sequence_builder, language_tokenizer, action_tokenizer):
        batch = {
            "task": {
                "language_instruction": ["Pick up the red block"]
            },
            "action": np.random.rand(1, 1, 10),  # Random action tensor
            "reasonings": ["First, I need to locate the red block"]
        }
        
        result = cot_sequence_builder.build_sequence(
            batch, language_tokenizer, action_tokenizer, begin_is_prompt=True
        )
        
        # Get the token ID for <begin_of_reasoning>
        bor_id = language_tokenizer.encode("<begin_of_reasoning>")[0]
        
        # Find where the <begin_of_reasoning> token is in the prompt
        prompt_tokens = result["prompt"]["tokens"][0]
        bor_positions = np.where(prompt_tokens == bor_id)[0]
        
        # Verify <begin_of_reasoning> is in prompt
        assert len(bor_positions) == 1, "Should have exactly one <begin_of_reasoning> token in prompt"
        bor_pos = bor_positions[0]
        
        # Verify it's marked as causal in mask_ar
        assert result["prompt"]["mask_ar"][0][bor_pos], "<begin_of_reasoning> should be marked as causal"
        
        # Verify it's not in generation sequence
        gen_tokens = result["gen"]["tokens"][0]
        assert bor_id not in gen_tokens, "<begin_of_reasoning> should not be in generation sequence"

    def test_pad_tokens_have_zero_mask(self, cot_sequence_builder, language_tokenizer, action_tokenizer):
        batch = {
            "task": {
                "language_instruction": ["hi"]  # Very short to ensure padding
            },
            "action": np.random.rand(1, 1, 10),
            "reasonings": ["ok"]  # Very short to ensure padding
        }
        
        result = cot_sequence_builder.build_sequence(
            batch, language_tokenizer, action_tokenizer
        )
        
        # Get pad token ID
        pad_id = language_tokenizer.encode("<pad>")[0]
        
        # Find pad tokens in prompt
        prompt_tokens = result["prompt"]["tokens"][0]
        pad_positions = np.where(prompt_tokens == pad_id)[0]
        
        # Verify all masks are zero for pad tokens in prompt
        assert len(pad_positions) > 0, "Test requires padding in prompt"
        assert not result["prompt"]["mask"][0][pad_positions].any(), "Mask should be zero for pad tokens"
        assert not result["prompt"]["mask_ar"][0][pad_positions].any(), "mask_ar should be zero for pad tokens"
        assert not result["prompt"]["mask_loss"][0][pad_positions].any(), "mask_loss should be zero for pad tokens"
        
        # Find pad tokens in generation
        gen_tokens = result["gen"]["tokens"][0]
        pad_positions = np.where(gen_tokens == pad_id)[0]
        
        # Verify all masks are zero for pad tokens in generation
        assert len(pad_positions) > 0, "Test requires padding in generation"
        assert not result["gen"]["mask"][0][pad_positions].any(), "Mask should be zero for pad tokens"
        assert not result["gen"]["mask_ar"][0][pad_positions].any(), "mask_ar should be zero for pad tokens"
        assert not result["gen"]["mask_loss"][0][pad_positions].any(), "mask_loss should be zero for pad tokens"
