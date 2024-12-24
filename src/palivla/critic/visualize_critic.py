import io
from typing import Dict

import jax
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from palivla.critic.model_components import CriticModelComponents
from palivla.typing import Data


def pad_to_batch(data: Data, *, batch_size: int) -> Data:
    def _pad_to_batch(x):
        return np.pad(x, ((0, batch_size - x.shape[0]),) + ((0, 0),) * (x.ndim - 1))

    return jax.tree.map(_pad_to_batch, data)


def apply_critic_to_trajectory(
    critic: CriticModelComponents, trajectory: Data, *, batch_size: int = 16
) -> Data:
    traj_len = trajectory["observation"]["image_primary"].shape[0]

    critic_values = []
    for i in range(0, traj_len, batch_size):
        batch = pad_to_batch(
            jax.tree.map(lambda x: x[i : i + batch_size], trajectory),
            batch_size=batch_size,
        )
        critic_value = critic.predict(batch)
        critic_values.append(critic_value)

    return np.concatenate(critic_values, axis=0)[:traj_len]


def visualize_critic(critic: CriticModelComponents, trajectory: Data) -> Dict:
    critic_values = apply_critic_to_trajectory(critic, trajectory)

    # Create figure with subplots
    fig = plt.figure(figsize=(10, 4), dpi=300)

    fig.suptitle(trajectory["task"]["language_instruction"][0].decode("utf-8"))

    num_plots = 8

    # Get total sequence length
    seq_len = trajectory["observation"]["image_primary"].shape[0]
    sample_indices = np.linspace(0, seq_len - 1, num_plots, dtype=int)

    # First row - 10 subsampled images
    for i, idx in enumerate(sample_indices):
        ax = plt.subplot(2, num_plots, i + 1)
        ax.imshow(trajectory["observation"]["image_primary"][idx])
        ax.set_title(f"t={idx}")
        ax.axis("off")

    # Second row - task_completed plot spanning full width
    ax = plt.subplot(2, 1, 2)
    ax.plot(critic_values)
    ax.set_xlabel("Time step")
    ax.set_ylabel("$Q(s, a)$")
    # ax.grid(True)

    plt.tight_layout()

    # Save to bytes
    img = io.BytesIO()
    plt.savefig(img, format="png")
    plt.close()

    img.seek(0)

    # Load from temp file
    img = Image.open(img)
    return img
