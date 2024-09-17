import math
from typing import List, Sequence, Tuple, Optional, Literal
from einops import einops, rearrange

import jax
import jax.numpy as jnp
import chex
import distrax
from flax import linen as nn
from flax.struct import dataclass


class FsqCodebook(nn.Module):
    input_dim: int
    num_tokens: int
    bins_per_dim: Sequence[int]

    @property
    def place_values(self):
        place_values = [1]
        for b in self.bins_per_dim[:-1]:
            place_values.append(place_values[-1] * b)
        return jnp.array(place_values)

    @classmethod
    def from_config(cls, input_dim, num_tokens, codebook_type, target_codebook_size):
        if codebook_type == "fsq":
            bins_per_dim = cls._get_bins_fsq(target_codebook_size)
        elif codebook_type == "lfq":
            bins_per_dim = cls._get_bins_lfq(target_codebook_size)
        else:
            raise ValueError(f"Codebook type {codebook_type} not supported.")

        return cls(
            input_dim=input_dim, num_tokens=num_tokens, bins_per_dim=bins_per_dim
        )

    @staticmethod
    def _get_bins_fsq(target_codebook_size):
        """
        Get bins per dimension based on codebook size, from the original FSQ paper.
        """
        if target_codebook_size == 2**8:
            return (8, 6, 5)
        elif target_codebook_size == 2**10:
            return (8, 5, 5, 5)
        elif target_codebook_size == 2**12:
            return (7, 5, 5, 5, 5)
        elif target_codebook_size == 2**14:
            return (8, 8, 8, 6, 5)
        elif target_codebook_size == 2**16:
            return (8, 8, 8, 5, 5, 5)
        else:
            raise ValueError(f"Codebook size {target_codebook_size} not supported.")

    @staticmethod
    def _get_bins_lfq(target_codebook_size):
        """
        Get bins per dimension according to the Lookup-Free Quantization paper (2 bins per dimension)
        """
        assert (
            target_codebook_size & (target_codebook_size - 1) == 0
        ), "Codebook size should be a power of two for LFQ"

        return (2,) * int(math.log2(target_codebook_size))

    def setup(self):
        self.proj_down = nn.Dense(len(self.bins_per_dim) * self.num_tokens)
        self.proj_up = nn.Dense(self.input_dim)

    def __call__(self, inputs):
        tokens, z = self.encode(inputs)
        output = self.decode(tokens, z_grad=z)
        return tokens, output

    def encode(self, inputs):
        bases = jnp.array(self.bins_per_dim)

        x = self.proj_down(inputs)
        x = rearrange(x, "... (n d) -> ... n d", n=self.num_tokens)
        z = jnp.tanh(x)

        # Quantize
        digits = jnp.round((z + 1) * (bases - 1) / 2).astype(jnp.int32)
        tokens = self.undigitize(digits)

        return tokens, z

    def decode(self, tokens, z_grad: Optional[jax.Array] = None):
        bases = jnp.array(self.bins_per_dim)
        digits = self.digitize(tokens)

        z_q = digits / (bases - 1) * 2 - 1

        if z_grad is not None:
            chex.assert_equal_shape([z_q, z_grad])
            z_q = jax.lax.stop_gradient(z_q - z_grad) + z_grad

        z_q = rearrange(z_q, "... n d -> ... (n d)")

        output = self.proj_up(z_q)

        return output

    def undigitize(self, digits):
        return jnp.sum(digits * jnp.array(self.place_values), axis=-1)

    def digitize(self, tokens):
        return (tokens[..., None] // jnp.array(self.place_values)) % jnp.array(
            self.bins_per_dim
        )

    @property
    def vocab_size(self):
        return math.prod(self.bins_per_dim)

class ResNetDownBlock(nn.Module):
    stride: int = 1
    n_filters: int = 64
    dropout_rate: float = 0.0
    group_size: int = 32

    @nn.compact
    def __call__(self, x, *, train=True):
        skip = x

        if self.stride > 1 or x.shape[-1] != self.n_filters:
            skip = nn.Conv(self.n_filters, (self.stride,), (self.stride,), "SAME")(skip)

        x = nn.Conv(self.n_filters, (3,), (self.stride,), "SAME")(x)
        x = nn.GroupNorm(num_groups=self.n_filters // self.group_size)(x)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=not train)
        x = nn.relu(x)
        x = nn.Conv(self.n_filters, (3,), (1,), "SAME")(x)

        return skip + x

class ResNetUpBlock(nn.Module):
    stride: int = 1
    n_filters: int = 64
    dropout_rate: float = 0.0
    group_size: int = 32

    @nn.compact
    def __call__(self, x, *, train=True):
        skip = x

        if self.stride > 1:
            skip = nn.ConvTranspose(self.n_filters, (self.stride,), (self.stride,), "SAME")(skip)

        x = nn.ConvTranspose(self.n_filters, (3,), (self.stride,), "SAME")(x)
        x = nn.GroupNorm(num_groups=self.n_filters // self.group_size)(x)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=not train)
        x = nn.relu(x)
        x = nn.ConvTranspose(self.n_filters, (3,), (1,), "SAME")(x)

        return skip + x


@dataclass
class LfqCodebookOutput:
    tokens: jnp.ndarray
    z: jnp.ndarray
    z_q: jnp.ndarray
    token_log_probs: jnp.ndarray
    commit_loss: jnp.ndarray


class LookupFreeQuantization(nn.Module):
    num_dims: int
    latent_dim: int

    def setup(self):
        self.codebook = jnp.array([-1, 1])
        # self.activation = lambda x: x
        self.activation = nn.tanh

        self.project_down = nn.Dense(self.num_dims)
        self.project_up = nn.Dense(self.latent_dim)

    def encode(self, z):
        z = self.project_down(z)
        token_squared_distances = jnp.square(z[..., None] - self.codebook)
        token_bits = jnp.argmin(token_squared_distances, axis=-1)
        tokens = jnp.sum(token_bits * (2 ** jnp.arange(self.num_dims)), axis=-1)
        return tokens

    def decode(self, tokens):
        token_bits = (tokens[..., None] & (2 ** jnp.arange(self.num_dims))).astype(jnp.int32)
        return self.project_up(self.codebook[token_bits])

    def loss(self, x):
        z = self.project_down(x)
        z = self.activation(z)

        token_squared_distances = jnp.square(z[..., None] - self.codebook)
        tokens = jnp.argmin(token_squared_distances, axis=-1)

        token_bit_log_probs = -token_squared_distances # jax.nn.log_softmax(-token_squared_distances, axis=-1)
        # Compute token log probs for tokens 0..2^num_dims-1 by summing corresponding log-probs
        token_bit_expansions = jnp.bitwise_and(
            jnp.arange(2 ** self.num_dims)[None, :],
            2 ** jnp.arange(self.num_dims)[:, None]
        ).astype(jnp.int32)
        token_log_probs = (
            token_bit_log_probs[..., 0] @ (1 - token_bit_expansions) +
            token_bit_log_probs[..., 1] @ token_bit_expansions
        ) # (batch_size, num_tokens, 2 ** num_dims)
        token_log_probs = jax.lax.stop_gradient(jax.nn.log_softmax(token_log_probs, axis=-1))
        chex.assert_shape(token_log_probs, (*x.shape[:-1], 2 ** self.num_dims))

        z_q = self.codebook[tokens]
        commit_loss = jnp.square(z - z_q).mean()
        z_q = jax.lax.stop_gradient(z_q - z) + z

        z_q = self.project_up(z_q)
        z = self.project_up(z)

        tokens = jnp.sum(tokens * (len(self.codebook) ** jnp.arange(self.num_dims)), axis=-1)
        return LfqCodebookOutput(
            tokens=tokens,
            z=z,
            z_q=z_q,
            token_log_probs=jnp.zeros(()),
            commit_loss=commit_loss,
        )


class LfqResnetTokenizer(nn.Module):
    stages: List[int]
    stage_filters: List[int]
    target_codebook_size: int
    data_dim: int
    data_horizon: int = 64

    l1_loss_weight: float = 0.0
    l2_loss_weight: float = 1.0
    commit_loss_weight: float = 0.0
    entropy_loss_weight: float = 0.05

    use_initial_offset: bool = False
    use_state_conditioning: bool = False

    @property
    def vocab_size(self):
        return self.target_codebook_size

    @property
    def num_tokens(self):
        return self.data_horizon // (2 ** len(self.stages))

    def setup(self):
        assert not self.use_state_conditioning

        encoder_layers = []

        encoder_layers.append(nn.Conv(self.stage_filters[0], (7,), (1,), "SAME"))
        for num_blocks, n_filters in zip(self.stages, self.stage_filters):
            for i in range(num_blocks):
                encoder_layers.append(ResNetDownBlock(
                    stride=2 if i == 0 else 1,
                    n_filters=n_filters,
                ))
        self.encoder_layers = encoder_layers

        decoder_layers = []
        for num_blocks, n_filters in zip(reversed(self.stages), reversed(self.stage_filters)):
            for i in range(num_blocks):
                decoder_layers.append(ResNetUpBlock(
                    stride=2 if i == 0 else 1,
                    n_filters=n_filters,
                ))
        decoder_layers.append(nn.ConvTranspose(self.data_dim, (7,), (1,), "SAME"))
        self.decoder_layers = decoder_layers

        self.codebook = LookupFreeQuantization(
            num_dims=int(math.log2(self.target_codebook_size)),
            latent_dim=self.stage_filters[-1],
        )

        if self.use_initial_offset:
            self.out_scale = self.param("out_scale", lambda _: jnp.full((), 1e-1))

    def _encoder(self, x, train=True):
        for layer in self.encoder_layers:
            if isinstance(layer, ResNetDownBlock):
                x = layer(x, train=train)
            else:
                x = layer(x)
        return x

    def _decoder(self, x, train=True):
        for layer in self.decoder_layers:
            if isinstance(layer, ResNetUpBlock):
                x = layer(x, train=train)
            else:
                x = layer(x)
        return x

    def tokenize(self, action, obs=None, train=True):
        x = self._encoder(action, train=train)
        tokens = self.codebook.encode(x)

        return tokens

    def detokenize(self, tokens, obs=None, train=True):
        x = self.codebook.decode(tokens)
        action = self._decoder(x, train=train) * self.out_scale

        return action

    def __call__(self, *args, **kwargs):
        """
        Dummy for .init
        """
        return self.loss(*args, **kwargs)

    def loss(self, action, obs=None, train=True, action_padding_mask=None, return_recon: bool = False):
        z = self._encoder(action, train=train)
        codebook_output = self.codebook.loss(z)
        action_recon = self._decoder(codebook_output.z_q, train=train) * self.out_scale

        # Reconstruction losses
        mse_loss = jnp.pow(action - action_recon, 2).mean()
        l1_loss = jnp.abs(action - action_recon).mean()

        # Commitment loss
        commit_loss = codebook_output.commit_loss

        # Compute entropy loss
        # token_log_probs = codebook_output.token_log_probs
        # token_probs = jnp.exp(token_log_probs)
        # all_but_last_two = tuple(range(token_probs.ndim - 2))
        # sample_entropy = -jnp.sum(jnp.where(token_probs > 1e-6, token_probs * token_log_probs, 0), axis=-1).mean()
        # token_marginal_probs = jnp.mean(token_probs, axis=all_but_last_two)
        # marginal_entropy = -jnp.sum(jnp.where(token_marginal_probs > 1e-6, token_marginal_probs * jnp.log(token_marginal_probs), 0), axis=-1).mean()
        entropy_loss = 0 # sample_entropy - marginal_entropy

        loss = (
            self.l2_loss_weight * mse_loss
            + self.l1_loss_weight * l1_loss
            + self.commit_loss_weight * commit_loss
            + self.entropy_loss_weight * entropy_loss
        )

        info = {
            "tokenizer_mse": mse_loss,
            "tokenizer_l1": l1_loss,
            "tokenizer_commit": commit_loss,
            "tokenizer_entropy": entropy_loss,
        }

        if return_recon:
            info["action_recon"] = action_recon

        return loss, info
