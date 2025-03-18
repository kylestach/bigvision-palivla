from typing import Any
import numpy as np
import wandb
import io

import jax
import jax.experimental.multihost_utils as mhu
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from big_vision.utils import Registry
from palivla.model_components import ModelComponents
from palivla.cot_parser import parse_cot_string, TrajectoryData


def get_table(lang_str: str, pred_cot: str, target_cot: str) -> wandb.Table:
    """Create a wandb table comparing the prompt and chain of thoughts."""
    table = wandb.Table(columns=["Prompt", "Predicted CoT", "GT CoT"])
    table.add_data(lang_str, pred_cot, target_cot)
    return table


def visualize_trajectory(
    ax: plt.Axes, image: np.ndarray, trajectory: TrajectoryData, title: str
) -> None:
    """Visualize a trajectory on a matplotlib axis.

    Args:
        ax: Matplotlib axis to draw on
        image: Input image to show
        trajectory: Parsed trajectory data containing gripper and object states
        title: Title for the plot
    """
    # Display the image
    ax.imshow(image.squeeze())

    # Scale factor from coordinate space to image space
    scale = 224.0 / 1024.0
    print(image.shape)

    legend = []

    # Plot gripper trajectory
    x = [state.x * scale for state in trajectory.gripper]
    y = [state.y * scale for state in trajectory.gripper]
    ax.plot(x, y, color="red", linewidth=2, alpha=0.3)
    for i, (x_, y_) in enumerate(zip(x, y)):
        ax.plot(
            x_,
            y_,
            marker="o",
            color="red",
            linestyle="-",
            linewidth=2,
            markersize=10,
            alpha=0.7**i,
            label="gripper" if i == 0 else None,
        )

    # Assign colors to objects
    colors = [
        "blue",
        "green",
        "orange",
        "purple",
        "pink",
        "brown",
        "gray",
        "olive",
        "cyan",
        "magenta",
    ]

    # Plot object bounding boxes
    for (obj_name, states), color in zip(trajectory.objects.items(), colors):
        for i, state in enumerate(states):
            x = state.x * scale
            y = state.y * scale
            width = state.width * scale if state.width else 0
            height = state.height * scale if state.height else 0

            rect = Rectangle(
                (x, y),
                width,
                height,
                linewidth=2,
                edgecolor=color,
                facecolor="none",
                alpha=0.7**i,
                label=obj_name if i == 0 else None,
            )
            ax.add_patch(rect)

    ax.legend()
    ax.set_title(title)
    ax.axis("off")


def create_side_by_side_visualization(
    image: np.ndarray,
    pred_trajectory: TrajectoryData,
    target_trajectory: TrajectoryData,
    prompt_str: str,
) -> np.ndarray:
    """Create a side-by-side visualization comparing predicted and target trajectories.

    Args:
        image: Input image to show in both panels
        pred_trajectory: Predicted trajectory from the model
        target_trajectory: Target/ground truth trajectory
        prompt_str: Text prompt to show as subtitle

    Returns:
        PIL Image containing the rendered figure
    """
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Remove <pad> from prompt string
    prompt_str = prompt_str.replace("<pad>", "")
    fig.suptitle(prompt_str, wrap=True)

    # Plot predicted and target trajectories
    visualize_trajectory(ax1, image, pred_trajectory, "Predicted")
    visualize_trajectory(ax2, image, target_trajectory, "Target")

    # Adjust layout and convert to image
    plt.tight_layout()

    # Convert figure to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)

    return img


@Registry.register("viz.chain_of_thought")
def chain_of_thought(model: ModelComponents, trajectory: Any):
    frame = jax.tree_map(lambda x: x[:1], trajectory)
    frame["observation"] = jax.tree_map(lambda x: x[None], frame["observation"])
    frame["action"] = trajectory["action"][None, :1, :]

    # Predict chain-of-thought
    sequences = model.build_sequence(frame, begin_is_prompt=True)
    viz_batch, sequences = mhu.broadcast_one_to_all(
        ({"observation": frame["observation"]}, sequences)
    )
    predicted_tokens = model.predict_tokens(
        viz_batch, sequences, use_ema_params=False, replicate=True
    )

    # Decode the tokens
    predicted_text_tokens = np.array(
        [model.language_tokenizer.decode(tok) for tok in predicted_tokens[0]]
    )
    target_gen_tokens = sequences["gen"]["tokens"]
    target_gen_text_tokens = np.array(
        [model.language_tokenizer.decode(tok) for tok in target_gen_tokens[0]]
    )
    prompt_tokens = sequences["prompt"]["tokens"]
    prompt_text_tokens = np.array(
        [
            model.language_tokenizer.decode(tok)
            for tok in prompt_tokens[0]
            if tok != "<pad>"
        ]
    )

    # Get action start indices
    pred_action_start_idxs = (
        np.argwhere(predicted_text_tokens == "<begin_of_action>") + 1
    )
    pred_action_start_idx = (
        np.min(pred_action_start_idxs) if pred_action_start_idxs.size > 0 else -1
    )
    target_action_start_idx = (
        np.min(np.argwhere(target_gen_text_tokens == "<begin_of_action>")) + 1
    )

    # Concatenate tokens into strings
    prompt_str = "".join(prompt_text_tokens)
    pred_cot_str = "".join(predicted_text_tokens[:pred_action_start_idx])
    target_cot_str = "".join(target_gen_text_tokens[:target_action_start_idx])

    # Plot the table of prompt, target reasoning str, gen reasoning str
    cot_table = get_table(prompt_str, pred_cot_str, target_cot_str)

    # Parse both predicted and target chain of thought strings
    pred_trajectory = parse_cot_string(pred_cot_str)
    target_trajectory = parse_cot_string(target_cot_str)

    # Create side-by-side visualization
    image = frame["observation"]["image_primary"].squeeze()
    comparison_image = create_side_by_side_visualization(
        image, pred_trajectory, target_trajectory, prompt_str
    )

    return {
        "trajectory_comparison": wandb.Image(comparison_image),
        "cot_outputs": cot_table,
    }
