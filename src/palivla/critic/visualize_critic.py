import io
from typing import Dict

import jax
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from jax.experimental import multihost_utils

from palivla.critic.model_components import CriticModelComponents
from palivla.typing import Data


def apply_critic_to_trajectory(
    critic: CriticModelComponents, trajectory: Data, *, batch_size: int = 16, num_counterfactual_actions: int = 16
) -> Data:
    # Use the first host's trajectory to compute the critic values.
    traj_len = trajectory["observation"]["image_primary"].shape[0]
    w0_traj_len = multihost_utils.broadcast_one_to_all(traj_len, is_source=(jax.process_index() == 0))

    # Pad the trajectory to a multiple of the batch size
    padded_traj_len = ((w0_traj_len - 1) // batch_size + 1) * batch_size
    pad_size = max(0, padded_traj_len - traj_len)
    def _pad_with_last(x):
        pad = np.repeat(x[-1:], pad_size, axis=0)
        return np.concatenate([x, pad], axis=0)[:padded_traj_len]
    trajectory = jax.tree.map(_pad_with_last, trajectory)

    critic_values = []
    baseline_values = []
    for i in range(0, padded_traj_len, batch_size):
        batch = jax.tree.map(lambda x: x[i : i + batch_size], trajectory).copy()
        batch["action"] = np.concatenate([
            batch["action"][:, None, :],
            np.random.normal(size=(batch_size, num_counterfactual_actions, batch["action"].shape[-1]))
        ], axis=1)
        critic_value = critic.predict(batch)

        qsa_value = critic_value[:, 0]
        random_action_value = critic_value[:, 1:].mean(axis=1)

        critic_values.append(qsa_value)
        baseline_values.append(random_action_value)

    return np.concatenate(critic_values, axis=0)[:traj_len], np.concatenate(baseline_values, axis=0)[:traj_len]


def visualize_critic(critic: CriticModelComponents, trajectory: Data) -> Dict:
    critic_values, baseline_values = apply_critic_to_trajectory(critic, trajectory)

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
    ax.plot(critic_values, label="Predicted $Q(s, a)$")
    ax.plot(baseline_values, label="Predicted $Q(s, a\sim\mathcal{N})$")
    ax.plot(trajectory["mc_return"], label="Monte-Carlo $Q(s, a)$")
    ax.set_xlabel("Time step")
    ax.set_ylabel("$Q(s, a)$")
    ax.legend()

    plt.tight_layout()

    # Save to bytes
    img = io.BytesIO()
    plt.savefig(img, format="png")
    plt.close()

    img.seek(0)

    # Load from temp file
    img = Image.open(img)
    return img
