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


@Registry.register("viz.sanity_print")
def sanity_print(model: ModelComponents, trajectory: Any):
    first_frame = jax.tree.map(lambda x: x[:1], trajectory)
    first_frame["observation"] = jax.tree.map(lambda x: x[None], first_frame["observation"])
    first_frame["action"] = trajectory["action"][None, :1, :]

    # Predict chain-of-thought
    sequences = model.build_sequence(first_frame, begin_is_prompt=True)
    predicted_tokens = model.predict_tokens(first_frame, sequences, use_ema_params=True)

    # Decode the tokens
    predicted_text_tokens = [model.language_tokenizer.decode(tok) for tok in predicted_tokens[0]]
    target_gen_tokens = sequences["gen"]["tokens"]
    target_gen_text_tokens = [model.language_tokenizer.decode(tok) for tok in target_gen_tokens[0]]
    prompt_tokens = sequences["prompt"]["tokens"]
    prompt_text_tokens = [model.language_tokenizer.decode(tok) for tok in prompt_tokens[0]]

    # Create figure with two columns, with ax2 taking up 3/4 of the width
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [1, 3]})
    
    # Plot image on left
    image = np.array(trajectory["observation"]["image_primary"][0])
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title("Input Image")

    # Make a table with two rows: text and token IDs
    table = prettytable.PrettyTable()
    def _add_row_shim(name, values):
        for i in range(0, len(values), row_length):
            row = values[i:i+row_length]
            pad_len = row_length - len(row)
            row += [""] * pad_len
            table.add_row([name if i == 0 else "", *row])

    row_length = 10
    _add_row_shim("Predicted Text", predicted_text_tokens)
    _add_row_shim("Predicted IDs", predicted_tokens[0].tolist())
    _add_row_shim("Target Text", target_gen_text_tokens)
    _add_row_shim("Target IDs", target_gen_tokens[0].tolist())

    # Plot text on right
    table_str = table.get_string(title='Sanity Print', header=False)
    # Compute font size based on column width
    column_width = len(table_str.split('\n')[0])
    num_rows = len(table_str.split('\n'))
    font_size = min(1000 / column_width, 500 / num_rows) # guess-and-check
    ax2.text(0.5, 0.5, table_str, wrap=True, family='monospace', fontsize=font_size, ha='center', va='center')
    ax2.axis('off')

    fig.suptitle("Prompt: " + ' :: '.join([t for t in prompt_text_tokens if t != "<pad>"]))

    # Adjust layout and return as wandb Image
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="jpeg")
    plt.close()
    buffer.seek(0)

    return {"sanity_print": wandb.Image(Image.open(buffer))}
