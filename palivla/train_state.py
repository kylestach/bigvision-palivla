from functools import partial
from flax.training.train_state import TrainState
from flax import struct
from flax.core.frozen_dict import FrozenDict
from scalax.sharding import MeshShardingHelper, ShardingRule
from flax import linen as nn
from typing import Any, Callable, Dict, Optional, Sequence, Tuple
import orbax.checkpoint as ocp
import jax
import jax.numpy as jnp
import dataclasses
from palivla.model import load_pretrained_params, get_model_spec, get_decode_fn
from palivla.tokenizer import BinActionTokenizer, Tokenizer
import tensorflow as tf
import flax
from tensorflow_text import SentencepieceTokenizer
from scalax.sharding import SJITCompiledFunction
from jax.sharding import PartitionSpec

import optax
from flax import linen as nn
from flax import nnx
import chex

from palivla.spec import ModuleSpec, OptimizerSpec
from palivla.train_step import step_fn
from palivla.types import Data, Info, Params


'''
def embed(encoders: Dict[str, nn.Module], data: Data, masks: Dict[str, jax.Array]):
    embeds = {}
    embed_masks = {}
    for k, enc in encoders.items():
        embeds[k], embed_masks[k] = enc(data[k], train=False)
    return embeds, embed_masks


def make_gather_indices(total_num_tokens: int, embeds_sizes: jnp.ndarray, reordering: jnp.ndarray):
    """
    Creates gather indices for reordering token groups.

    Args:
        embeds_sizes: Array of shape (batch_size, num_token_groups) containing the number of tokens in each group.
        reordering: Array of shape (batch_size, num_token_groups) containing the new order of token groups.

    Returns:
        gather_indices: Array of shape (batch_size, total_num_tokens) containing the indices to gather tokens in the new order.

    Example:
        total_num_tokens = 8
        embeds_sizes = [3, 4, 1]  # text: 3, image: 4, proprio: 1
        reordering = [[2, 1, 0]]    # new order: proprio, image, text
        Result: [7, 3, 4, 5, 6, 0, 1, 2]
    
    Method:
     - First, find the start and end of each token group in the original order and in the new order.
     - For each token group find the delta between the start of the token group in the new order and the start of the token group in the original order.
     - For each index in the output, which token group it belongs to in the new order, and what is the offset delta of this group.
     - The output indices are then the original indices of the tokens in the token group, plus the delta of the token group.
    """
    inverse_reordering = jnp.argsort(reordering)

    # Find the embed sizes, reordered for the new token group order
    embeds_sizes_reordered = embeds_sizes[reordering]

    # Calculate cumulative sizes for original and reordered embeddings
    cumulative_sizes = jnp.cumsum(embeds_sizes)
    cumulative_sizes_reordered = jnp.cumsum(embeds_sizes_reordered)

    # Pad with zeros at the beginning for easier indexing
    cumulative_sizes = jnp.pad(cumulative_sizes, ((1, 0)))
    cumulative_sizes_reordered = jnp.pad(cumulative_sizes_reordered, ((1, 0)))

    offsets_original = cumulative_sizes[:-1]
    offsets_reordered = cumulative_sizes_reordered[:-1][inverse_reordering]

    offset_delta = offsets_original - offsets_reordered

    # Find which group each token belongs to in the reordered sequence
    membership_in_reordered_groups = jnp.sum(jnp.arange(total_num_tokens)[:, None] >= cumulative_sizes_reordered[None, :], axis=-1) - 1
    membership_in_original_groups = reordering[membership_in_reordered_groups]

    return jnp.arange(total_num_tokens) + offset_delta[membership_in_original_groups], membership_in_original_groups


def pack_embeddings(embeds: jax.Array, masks: jax.Array):
    idcs = jnp.argsort(~masks)
    jax.debug.print("{}", masks)
    jax.debug.print("{}", idcs)
    return embeds[idcs]


def collect_embeddings(embeds: Dict[str, jax.Array], embed_masks: Dict[str, jax.Array], language_embeds: jax.Array, language_masks: jax.Array, rng: jax.Array):
    """
    Collects embeddings for a single sample (with no batch axis), randomly reordered and packed according to masks.
    """
    # Get a consistent order to use throughout this function.
    key_order = list(sorted(embeds.keys())) + ["llm"]

    # Number of tokens in each embed.
    embed_sizes = [embeds[k].shape[0] for k in key_order]
    total_num_tokens = sum(embed_sizes)

    # Make embeds into a single contiguous array in the original order
    concat_embeds = jnp.concatenate([embeds[k] for k in key_order], axis=0)
    concat_masks = jnp.concatenate([embed_masks[k] for k in key_order], axis=0)

    # Randomly reorder the embeddings.
    # The last token group is the language tokens, which we want to attend to with full self-attention.
    sequence_order = jnp.concatenate([jax.random.permutation(rng, len(embeds) - 1), [len(embeds) - 1]])
    gather_indices, membership_in_original_groups = make_gather_indices(total_num_tokens, jnp.array(embed_sizes), jnp.array(sequence_order))

    # Pack the embeddings according to the masks, and compute the start indices as the first unmasked index in the packed sequence.
    packed_embeds = pack_embeddings(concat_embeds[gather_indices], concat_masks[gather_indices])
    packed_masks = pack_embeddings(concat_masks[gather_indices], concat_masks[gather_indices])
    start_indices = jnp.argmax(~packed_masks, axis=0)

    # Make the mask for which tokens should be autoregressively attended to.
    mask_ar = membership_in_original_groups == len(embeds) - 1

    return packed_embeds, packed_masks, start_indices, mask_ar


import importlib
from big_vision.models.proj.paligemma import paligemma

class PaliVLAModel(paligemma.Model):
    encoder_models: Optional[FrozenDict[str, str]] = None
    encoder_configs: Optional[FrozenDict[str, FrozenDict]] = None

    def setup(self):
        self._llm = importlib.import_module(
            f"big_vision.models.{self.llm_model}"
        ).Model(**(self.llm or {}), name="llm")
        self._img = importlib.import_module(
            f"big_vision.models.{self.img_model}"
        ).Model(**self.img_config, name="img")

        for modality in self.encoder_models.keys():
            encoder_model = self.encoder_models[modality]
            encoder_config = self.encoder_configs[modality]

            encoder = importlib.import_module(
                f"big_vision.models.{encoder_model}"
            ).Model(**encoder_config, name=modality)
            setattr(self, "_" + modality, encoder)
        
        self.encoders = {modality: getattr(self, "_" + modality) for modality in list(self.encoder_models.keys()) + ["_img"]}

    def __call__(self, observations, observations_masks, text, text_masks, mask_ar, train=False):
        text_embeds = self._llm.embed_tokens(text, train=train)

        embeds, embed_masks = embed(self.encoders, observations, observations_masks)
        packed_embeds, packed_masks, start_indices, mask_ar = collect_embeddings(embeds, embed_masks, text_embeds, text_masks, self.make_rng("dropout"))
        attn_mask = paligemma.make_attn_mask(packed_masks, mask_ar)
        _, out_llm = self._llm(packed_embeds, mask=attn_mask, train=train)

        # Pad the logits to the original text sequence length.
        embeddings_pad = jnp.pad(out_llm["pre_logits"], ((0, 0), (0, text.shape[1]), (0, 0)))
        text_masks = jnp.pad(text_masks, ((0, 0), (0, text.shape[1])))
        embeddings_text = jax.lax.dynamic_slice(embeddings_pad, (0, start_indices, 0), (text.shape[0], text.shape[1], -1))

        return self._llm.compute_logits(embeddings_text, train=train)
    
    def prefill_cache(self, x, input_mask, mask_ar, *, cache_size):
        embeds, embed_masks = embed(self.encoders, x, input_mask)
        packed_embeds, packed_masks, start_indices, mask_ar = collect_embeddings(embeds, embed_masks, text_embeds, text_masks, self.make_rng("dropout"))
'''


def restore_gluon_module(path: str, mesh: MeshShardingHelper, step: int | None = None, extra_kwargs: Dict[str, Any] = {}):
    with tf.io.gfile.GFile(tf.io.gfile.join(path, "module_spec.json"), "r") as f:
        module_spec = ModuleSpec.from_json(f.read())

    module_spec.config.update(extra_kwargs)

    params_manager = ocp.CheckpointManager(
        directory=tf.io.gfile.join(path, "checkpoints"),
        item_handlers={"default": ocp.StandardCheckpointHandler()},
    )

    params_metadata = params_manager.item_metadata(params_manager.latest_step())[
        "default"
    ]
    sharding = jax.sharding.NamedSharding(mesh, PartitionSpec())
    abstract_params = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=sharding), params_metadata
    )
    params = params_manager.restore(
        params_manager.latest_step(),
        args=ocp.args.Composite(default=ocp.args.StandardRestore(abstract_params)),
    )["default"]["params"]
    return module_spec, params


class PaliVLA:
    train_state: TrainState

    module_spec: ModuleSpec
    optimizer_spec: Optional[OptimizerSpec]
    action_tokenizer_spec: Optional[ModuleSpec]

    model_sharding: Optional[ShardingRule]
    data_sharding: Optional[ShardingRule]
    mesh: MeshShardingHelper
    tokenizer: Tokenizer

    dataset_statistics: Dict[str, Any]

    module: nn.Module

    rng: jnp.ndarray

    decode_fn: Callable[[Params, Data, nn.Module], jax.Array]

    checkpoint_manager: Optional[ocp.CheckpointManager]

    image_keys: Sequence[str] = ("primary",)

    step_fn: Optional[SJITCompiledFunction] = None

    def __init__(
        self,
        *,
        train_state: TrainState,
        module_spec: ModuleSpec,
        optimizer_spec: OptimizerSpec,
        action_tokenizer_spec: ModuleSpec,
        model_sharding: ShardingRule,
        data_sharding: ShardingRule,
        mesh: MeshShardingHelper,
        tokenizer: Tokenizer,
        dataset_statistics: Dict[str, Any],
        module: nn.Module,
        rng: jnp.ndarray,
        decode_fn: Callable[[Params, Data, nn.Module], jax.Array],
        checkpoint_manager: Optional[ocp.CheckpointManager],
        image_keys: Sequence[str] = ("primary",),
    ):
        self.train_state = train_state
        self.module_spec = module_spec
        self.optimizer_spec = optimizer_spec
        self.action_tokenizer_spec = action_tokenizer_spec
        self.model_sharding = model_sharding
        self.data_sharding = data_sharding
        self.mesh = mesh
        self.tokenizer = tokenizer
        self.dataset_statistics = dataset_statistics
        self.module = module
        self.rng = rng
        self.decode_fn = decode_fn
        self.checkpoint_manager = checkpoint_manager
        self.step_fn = (
            self.mesh.sjit(
                step_fn,
                static_argnums=(3, 4),
                in_shardings=(self.model_sharding, self.data_sharding, None, None),
                out_shardings=(self.model_sharding, None, None),
                args_sharding_constraint=(
                    self.model_sharding,
                    self.data_sharding,
                    None,
                    None,
                ),
                donate_argnums=(0,),
            )
            if mesh is not None
            else None
        )
        self.image_keys = image_keys

    @classmethod
    def from_pretrained(
        cls,
        model_load_path: str,
        language_tokenizer_path: str,
        num_proprio_tokens: int = 0,
        proprio_dim: int = 7,
        action_dim: int = 7,
        action_tokenizer_path: Optional[str] = None,
        prompt_autoregressive: bool = False,
        optimizer_spec: Optional[OptimizerSpec] = None,
        mesh: Optional[MeshShardingHelper] = None,
        model_sharding: Optional[ShardingRule] = None,
        data_sharding: Optional[ShardingRule] = None,
        model_dtype: jnp.dtype = jnp.float32,
        dataset_statistics: Optional[Dict[str, Any]] = None,
        rng_seed: Optional[int] = 0,
        checkpoint_save_path: Optional[str] = None,
        image_keys: Sequence[str] = ("image_primary",),
    ):
        if mesh is None:
            mesh = MeshShardingHelper([-1], ["fsdp"])

        # Load language tokenizer
        with tf.io.gfile.GFile(language_tokenizer_path, "rb") as f:
            language_tokenizer = SentencepieceTokenizer(f.read())

        # Load action tokenizer
        if action_tokenizer_path is None:
            action_tokenizer_spec = ModuleSpec(BinActionTokenizer, {
                "action_vocab_offset": 256000,
                "action_vocab_size": 256,
                "action_dim": action_dim,
                "min_action_value": -2,
                "max_action_value": 2,
            })
            action_tokenizer_params = {}
        else:
            action_tokenizer_spec, action_tokenizer_params = restore_gluon_module(
                action_tokenizer_path,
                mesh=mesh.mesh,
            )

        action_tokenizer_params = mesh.apply_shard_and_gather_fns(mesh.make_shard_and_gather_fns(action_tokenizer_params, PartitionSpec())[0], action_tokenizer_params)

        action_tokenizer = action_tokenizer_spec.instantiate()

        tokenizer = Tokenizer.from_components(
            language_tokenizer,
            action_tokenizer,
            action_tokenizer_params,
            prompt_autoregressive=prompt_autoregressive,
        )

        rng = jax.random.PRNGKey(rng_seed)
        rng, init_rng = jax.random.split(rng)

        model_spec = get_model_spec(num_proprio_tokens=num_proprio_tokens)
        model = model_spec.instantiate()
        model_params = load_pretrained_params(
            model_load_path,
            dtype=model_dtype,
        )
        model_params["proprio"] = model.init(init_rng, jnp.zeros((1, proprio_dim)), method="embed_proprio")["params"]["proprio"]

        decode_fn = get_decode_fn(model, tokenizer)

        if optimizer_spec is None:
            optimizer_spec = OptimizerSpec(optax.set_to_zero, {})
        tx = optimizer_spec.instantiate()

        if dataset_statistics is None:
            dataset_statistics = {}

        @partial(
            mesh.sjit,
            in_shardings=None,
            out_shardings=model_sharding,
            donate_argnums=(0,),
        )
        def init_fn(params):
            return TrainState.create(
                apply_fn=model.apply,
                params=params,
                tx=tx,
            )

        train_state = init_fn(model_params)

        if checkpoint_save_path is not None:
            checkpoint_manager = ocp.CheckpointManager(
                directory=checkpoint_save_path,
                item_names=[
                    "params",
                    "opt_state",
                    "step",
                    "model_spec",
                    "optimizer_spec",
                    "dataset_statistics",
                    "tokenizer_config",
                    "rng",
                    "action_tokenizer_spec",
                    "action_tokenizer_params",
                ],
                item_handlers={
                    "params": ocp.StandardCheckpointHandler(),
                    "opt_state": ocp.StandardCheckpointHandler(),
                    "step": ocp.StandardCheckpointHandler(),
                    "model_spec": ocp.JsonCheckpointHandler(),
                    "optimizer_spec": ocp.JsonCheckpointHandler(),
                    "dataset_statistics": ocp.JsonCheckpointHandler(),
                    "tokenizer_config": ocp.JsonCheckpointHandler(),
                    "rng": ocp.JaxRandomKeyCheckpointHandler(),
                    "action_tokenizer_spec": ocp.JsonCheckpointHandler(),
                    "action_tokenizer_params": ocp.StandardCheckpointHandler(),
                },
                options=ocp.CheckpointManagerOptions(
                    max_to_keep=1,
                ),
            )
        else:
            checkpoint_manager = None

        return cls(
            train_state=train_state,
            module_spec=model_spec,
            optimizer_spec=optimizer_spec,
            action_tokenizer_spec=action_tokenizer_spec,
            model_sharding=model_sharding,
            data_sharding=data_sharding,
            mesh=mesh,
            tokenizer=tokenizer,
            dataset_statistics=dataset_statistics,
            module=model,
            rng=rng,
            decode_fn=decode_fn,
            checkpoint_manager=checkpoint_manager,
            image_keys=image_keys,
        )

    def save(self, step: int):
        args=ocp.args.Composite(
            params=ocp.args.StandardSave(self.train_state.params),
            opt_state=ocp.args.StandardSave(self.train_state.opt_state),
            # step=ocp.args.StandardSave({"step": self.train_state.step}),
            # rng=ocp.args.JaxRandomKeySave(self.rng),
            model_spec=ocp.args.JsonSave(self.module_spec.to_dict()),
            optimizer_spec=ocp.args.JsonSave(self.optimizer_spec.to_dict()),
            dataset_statistics=ocp.args.JsonSave(self.dataset_statistics),
            # tokenizer_config=ocp.args.JsonSave(
            #     dataclasses.asdict(self.tokenizer.config)
            # ),
            # action_tokenizer_spec=ocp.args.JsonSave(
            #     self.action_tokenizer_spec.to_dict()
            # ),
            # action_tokenizer_params=ocp.args.StandardSave(
            #     self.tokenizer.action_tokenizer_params
            # ),
        )

        self.checkpoint_manager.save(
            step,
            args=args,
        )

    @classmethod
    def from_checkpoint(
        cls,
        load_directory: str,
        language_tokenizer_path: str,
        step: int,
        save_directory: Optional[str] = None,
        mesh: Optional[MeshShardingHelper] = None,
        model_sharding: Optional[ShardingRule] = None,
        data_sharding: Optional[ShardingRule] = None,
        load_optimizer: bool = True,
        model_dtype: Optional[jnp.dtype] = None,
        action_tokenizer_path: Optional[str] = None,
    ):
        load_checkpoint_manager = ocp.CheckpointManager(
            directory=load_directory,
            item_names=[
                "params",
                "opt_state",
                "step",
                "model_spec",
                "optimizer_spec",
                "dataset_statistics",
                "tokenizer_config",
                "rng",
                "action_tokenizer_spec",
                "action_tokenizer_params",
            ],
            item_handlers={
                "params": ocp.StandardCheckpointHandler(),
                "opt_state": ocp.StandardCheckpointHandler(),
                "step": ocp.StandardCheckpointHandler(),
                "model_spec": ocp.JsonCheckpointHandler(),
                "optimizer_spec": ocp.JsonCheckpointHandler(),
                "dataset_statistics": ocp.JsonCheckpointHandler(),
                "tokenizer_config": ocp.JsonCheckpointHandler(),
                "rng": ocp.JaxRandomKeyCheckpointHandler(),
                "action_tokenizer_spec": ocp.JsonCheckpointHandler(),
                "action_tokenizer_params": ocp.StandardCheckpointHandler(),
            },
            options=ocp.CheckpointManagerOptions(),
        )

        # Restore JSON saved objects
        rng = jax.random.PRNGKey(0)
        train_step_sharding = jax.sharding.NamedSharding(
            mesh.mesh,
            model_sharding.apply(jax.ShapeDtypeStruct((), jnp.int64)) if isinstance(model_sharding, ShardingRule) else model_sharding,
        )

        restored_metadata = load_checkpoint_manager.restore(
            step,
            args=ocp.args.Composite(
                # step=ocp.args.StandardRestore(
                #     {"step": jax.ShapeDtypeStruct((), jnp.int64, sharding=train_step_sharding)}
                # ),
                model_spec=ocp.args.JsonRestore(),
                optimizer_spec=ocp.args.JsonRestore(),
                dataset_statistics=ocp.args.JsonRestore(),
                rng=ocp.args.JaxRandomKeyRestore(rng),
                tokenizer_config=ocp.args.JsonRestore(),
                action_tokenizer_spec=ocp.args.JsonRestore(),
                action_tokenizer_params=ocp.args.StandardRestore(
                    jax.tree.map(
                        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
                        load_checkpoint_manager.item_metadata(step)[
                            "action_tokenizer_params"
                        ],
                    )
                ),
            ),
        )
        rng = restored_metadata["rng"]
        # print(restored_metadata["step"])
        # train_step = restored_metadata["step"]
        train_step = step
        model_spec = ModuleSpec.from_dict(restored_metadata["model_spec"])
        optimizer_spec = OptimizerSpec.from_dict(restored_metadata["optimizer_spec"])
        # action_tokenizer_spec = ModuleSpec.from_dict(
        #     restored_metadata["action_tokenizer_spec"]
        # )
        action_tokenizer_spec, action_tokenizer_params = restore_gluon_module(
            action_tokenizer_path,
            mesh=mesh.mesh,
        )

        dataset_statistics = restored_metadata["dataset_statistics"]
        with tf.io.gfile.GFile(language_tokenizer_path, "rb") as f:
            language_tokenizer = SentencepieceTokenizer(f.read())

        tokenizer = Tokenizer.from_components(
            language_tokenizer=language_tokenizer,
            action_tokenizer=action_tokenizer_spec.instantiate(),
            action_tokenizer_params=action_tokenizer_params,
            prompt_autoregressive=False,
            #restored_metadata["tokenizer_config"][
            #     "prompt_autoregressive"
            # ],
        )

        # Load sharded params
        params_metadata = load_checkpoint_manager.item_metadata(step)["params"]

        def _make_abstract_array(metadata):
            return jax.ShapeDtypeStruct(
                shape=metadata.shape,
                dtype=metadata.dtype,
            )

        def _shard_abstract_array(abstract_array, sharding_rule):
            return jax.ShapeDtypeStruct(
                shape=abstract_array.shape,
                dtype=model_dtype or abstract_array.dtype,
                sharding=sharding_rule,
            )

        abstract_params = jax.tree_map(_make_abstract_array, params_metadata)
        if isinstance(model_sharding, ShardingRule):
            params_sharding_rules = model_sharding.apply(abstract_params)
        else:
            params_sharding_rules = jax.tree_map(lambda _: model_sharding, abstract_params)
        params_sharding_rules = jax.tree.map(lambda p: jax.NamedSharding(mesh.mesh, p), params_sharding_rules)

        abstract_params = jax.tree.map(
            _shard_abstract_array,
            abstract_params,
            params_sharding_rules,
        )
        params = load_checkpoint_manager.restore(
            step,
            args=ocp.args.Composite(params=ocp.args.StandardRestore(abstract_params)),
        )["params"]

        if load_optimizer:
            tx = optimizer_spec.instantiate()
            abstract_optimizer_state = jax.eval_shape(tx.init, abstract_params)
            optimizer_sharding_rules = model_sharding.apply(abstract_optimizer_state)
            abstract_optimizer_state = jax.tree.map(
                _shard_abstract_array,
                abstract_optimizer_state,
                optimizer_sharding_rules,
            )
            opt_state = load_checkpoint_manager.restore(
                step,
                args=ocp.args.Composite(
                    opt_state=ocp.args.StandardRestore(abstract_optimizer_state)
                ),
            )["opt_state"]
        else:
            optimizer_spec = OptimizerSpec(optax.set_to_zero, {})
            opt_state = {}

        model = model_spec.instantiate()
        tx = optimizer_spec.instantiate()
        decode_fn = get_decode_fn(model, tokenizer)

        if save_directory is not None:
            checkpoint_manager = ocp.CheckpointManager(
                directory=save_directory,
                item_names=[
                    "params",
                    "opt_state",
                    "step",
                    "model_spec",
                    "optimizer_spec",
                    "dataset_statistics",
                    "tokenizer_config",
                    "rng",
                ],
                options=ocp.CheckpointManagerOptions(
                    max_to_keep=1,
                ),
            )
        else:
            checkpoint_manager = None

        return cls(
            train_state=TrainState(
                step=train_step,
                apply_fn=model.apply,
                params=params,
                tx=tx,
                opt_state=opt_state,
            ),
            module_spec=model_spec,
            optimizer_spec=optimizer_spec,
            dataset_statistics=dataset_statistics,
            tokenizer=tokenizer,
            module=model_spec.instantiate(),
            model_sharding=model_sharding,
            data_sharding=data_sharding,
            mesh=mesh,
            decode_fn=decode_fn,
            checkpoint_manager=checkpoint_manager,
            rng=rng,
            action_tokenizer_spec=action_tokenizer_spec,
        )

    def replace(self, **updates) -> "TrainState":
        return dataclasses.replace(self, **updates)

    def process_image(self, obs: Dict[str, jax.Array]) -> jax.Array:
        # Concatenate and normalize images
        return jnp.concatenate([obs[image_key] for image_key in self.image_keys], axis=1) / 127.5 - 1.0

    def train_step(self, batch: Data):
        batch = self.mesh.local_data_to_global_array(batch)

        with self.mesh.mesh, nn.logical_axis_rules([("act_batch", "fsdp")]):
            self.train_state, info, self.rng = self.step_fn(
                self.train_state, batch, self.rng, self.tokenizer.config, self.tokenizer.action_tokenizer, self.tokenizer.action_tokenizer_params
            )

        return info

    def decode(
        self,
        image: jax.Array,
        prompt: jax.Array,
        mask_ar: jax.Array,
        mask_input: jax.Array,
        proprio: jax.Array | None = None,
    ) -> Tuple[jnp.ndarray, Info]:
        batch = {
            "image": image,
            "text": prompt,
            "mask_ar": mask_ar,
            "mask_input": mask_input,
            "proprio": proprio,
            "_mask": jnp.ones_like(prompt[:, 0]),
        }
        return self.decode_fn(
            {"params": self.train_state.params},
            batch,
            model=self.module,
            devices=self.mesh.mesh.devices,
            max_decode_len=self.tokenizer.config.num_action_tokens,
            replicate_out=True,
            mesh=self.mesh.mesh,
        )
