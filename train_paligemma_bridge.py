import collections
from functools import partial
import os
from pathlib import Path
from pprint import pprint

import jax
import chex
import jax.numpy as jnp
import ml_collections
import optax
import tensorflow as tf
import tqdm
from absl import app, flags
from flax import struct
from flax.training.train_state import TrainState
from ml_collections import ConfigDict, config_flags
from tensorflow_text import SentencepieceTokenizer
from scalax.sharding import (
    MeshShardingHelper,
    FSDPShardingRule,
    PartitionSpec,
    NamedSharding,
)
import jax_smi
import wandb
import numpy as np
from jax.experimental import multihost_utils
import orbax.checkpoint as ocp

import tracemalloc
import linecache
                
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
tf.config.set_visible_devices([], "GPU")


@struct.dataclass
class Tokenizer:
    @struct.dataclass
    class TokenizerConfig:
        action_vocab_size: int = struct.field(pytree_node=False)
        action_vocab_offset: int = struct.field(pytree_node=False)
        vocab_size: int = struct.field(pytree_node=False)
        num_action_tokens: int = struct.field(pytree_node=False)
        bos_token: int = struct.field(pytree_node=True)
        eos_token: int = struct.field(pytree_node=True)
        pad_token: int = struct.field(pytree_node=True)
        begin_of_action_token: int = struct.field(pytree_node=True)
        max_pad_length: int = struct.field(pytree_node=False)
        min_action_value: float = struct.field(pytree_node=True)
        max_action_value: float = struct.field(pytree_node=True)

        def bin_tokenize(self, data):
            # Assume normalization and clipping to [-1, 1]
            data = tf.clip_by_value(data, self.min_action_value, self.max_action_value)
            data = (data - self.min_action_value) / (
                self.max_action_value - self.min_action_value
            )
            return (
                tf.clip_by_value(
                    tf.cast(data * self.action_vocab_size, tf.int32),
                    0,
                    self.action_vocab_size - 1,
                )
                + self.action_vocab_offset
            )

        def bin_detokenize(self, tokens):
            _np = jax.numpy if isinstance(tokens, jax.Array) else np
            values = (tokens - self.action_vocab_offset) / self.action_vocab_size
            values = _np.where((values < 0) | (values > 1), _np.nan, values)
            data = (
                values * (self.max_action_value - self.min_action_value)
                + self.min_action_value
            )
            return data

    config: ConfigDict
    language_tokenizer: SentencepieceTokenizer
    token_structure: dict = struct.field(pytree_node=False)

    @classmethod
    def from_tokenizer(cls, tokenizer: SentencepieceTokenizer):
        bos_token = tokenizer.string_to_id("<bos>").numpy().item()
        eos_token = tokenizer.string_to_id("<eos>").numpy().item()
        pad_token = tokenizer.string_to_id("<pad>").numpy().item()
        begin_of_action_token = tokenizer.string_to_id("\n").numpy().item()
        max_pad_length = 60

        return cls(
            config=cls.TokenizerConfig(
                action_vocab_size=256,
                action_vocab_offset=256000,
                num_action_tokens=7,
                bos_token=bos_token,
                eos_token=eos_token,
                pad_token=pad_token,
                begin_of_action_token=begin_of_action_token,
                max_pad_length=max_pad_length,
                min_action_value=-2,
                max_action_value=2,
                vocab_size=tokenizer.vocab_size().numpy().item(),
            ),
            language_tokenizer=tokenizer,
            token_structure={
                "prefix": [
                    [bos_token],
                    "prompt",
                    [begin_of_action_token],
                ],
                "causal": [
                    "action",
                ],
                "pad": [[pad_token] * max_pad_length],
            },
        )

    def compose_token_structure(self, tokens, include_keys=["prefix", "causal", "pad"]):
        def _extract_tokens(ids_or_str):
            if isinstance(ids_or_str, str):
                return tokens[ids_or_str]
            else:
                return tf.constant(ids_or_str, dtype=tf.int32)

        tokens_by_name = {
            k: (
                tf.concat([_extract_tokens(token) for token in v], axis=0)
                if k in include_keys
                else tf.zeros((0,), dtype=tf.int32)
            )
            for k, v in self.token_structure.items()
        }

        tokens = tf.concat(
            [tokens_by_name["prefix"], tokens_by_name["causal"], tokens_by_name["pad"]],
            axis=0,
        )[: self.config.max_pad_length]
        mask_ar = tf.concat(
            [
                tf.zeros_like(tokens_by_name["prefix"], dtype=tf.bool),
                tf.ones_like(tokens_by_name["causal"], dtype=tf.bool),
                tf.ones_like(tokens_by_name["pad"], dtype=tf.bool),
            ],
            axis=0,
        )[: self.config.max_pad_length]
        mask_loss = tf.concat(
            [
                tf.zeros_like(tokens_by_name["prefix"], dtype=tf.bool),
                tf.ones_like(tokens_by_name["causal"], dtype=tf.bool),
                tf.zeros_like(tokens_by_name["pad"], dtype=tf.bool),
            ],
            axis=0,
        )[: self.config.max_pad_length]

        return tokens, mask_ar, mask_loss

    def tokenize_language_instruction(self, data):
        instruction = data["task"]["language_instruction"]
        instruction = tf.strings.lower(instruction)
        instruction = tf.strings.regex_replace(instruction, "[.?!]", "")
        instruction = tf.strings.regex_replace(instruction, "\n", " ")
        instruction = tf.strings.strip(instruction)
        instruction = tf.strings.join([tf.constant("act "), instruction])

        data["language_instruction_tokens"] = self.language_tokenizer.tokenize(
            instruction
        )

        return data

    def bin_tokenize(self, data):
        return self.config.bin_tokenize(data)

    def bin_detokenize(self, tokens):
        return self.config.bin_detokenize(tokens)

    def prepare_tokens_for_training(self, data):
        tokens = {
            "prompt": data["language_instruction_tokens"][
                : self.config.max_pad_length - 10
            ],
            "action": self.bin_tokenize(tf.squeeze(data["action"], axis=(0, 1))),
        }

        tokens, mask_ar, mask_loss = self.compose_token_structure(tokens)

        data["tokens"] = tokens
        data["mask_ar"] = mask_ar
        data["mask_loss"] = mask_loss
        data["mask_input"] = tokens != self.config.pad_token

        del data["language_instruction_tokens"]

        return data

    def prepare_tokens_for_generation(self, data):
        tokens = {
            "prompt": data["language_instruction_tokens"][
                : self.config.max_pad_length - 10
            ],
        }

        tokens, mask_ar, mask_loss = self.compose_token_structure(
            tokens, include_keys={"prefix", "pad"}
        )

        del data["language_instruction_tokens"]

        data["tokens"] = tokens
        data["mask_ar"] = mask_ar
        data["mask_input"] = tokens != self.config.pad_token

        return data

    def extract_action(self, data):
        action_start = (
            jnp.argmax(data["tokens"] == self.config.begin_of_action_token, axis=-1) + 1
        )
        action_data = data["tokens"][
            action_start : action_start + self.config.num_action_tokens
        ]
        action_data = self.bin_detokenize(action_data)
        return action_data


def prepare_image(data):
    data["observation"]["image_primary"] = (
        tf.cast(data["observation"]["image_primary"], tf.float32) / 127.5 - 1
    )
    return data


def load_model(config: ConfigDict, tokenizer: SentencepieceTokenizer):
    from big_vision.models.proj.paligemma import paligemma
    from big_vision.trainers.proj.paligemma import predict_fns

    # Define model
    model_config = ml_collections.FrozenConfigDict(
        {
            "llm": {"vocab_size": 257_152},
            "img": {
                "variant": "So400m/14",
                "pool_type": "none",
                "scan": True,
            },
        }
    )
    model = paligemma.Model(**model_config)

    # Load params - this can take up to 1 minute in T4 colabs.
    params = paligemma.load(None, config.model_path, model_config)

    # Change params to fp32
    params = jax.tree.map(lambda x: x.astype(jnp.float32), params)

    decode_fn = predict_fns.get_all(model)["decode"]
    decode = partial(
        decode_fn,
        devices=jax.devices(),
        eos_token=tokenizer.config.eos_token,
    )

    return model, params, decode


def make_dataset(config: ConfigDict, tokenizer: Tokenizer, train: bool):
    from octo.data.dataset import make_interleaved_dataset
    from octo.data.oxe import make_oxe_dataset_kwargs_and_weights

    dataset_kwargs = config.dataset_kwargs.to_dict()

    dataset_kwargs["dataset_kwargs_list"], dataset_kwargs["sample_weights"] = (
        make_oxe_dataset_kwargs_and_weights(**dataset_kwargs.pop("oxe_kwargs"))
    )

    # For now, always test on the training split since train rollout MSE is more informative
    dataset = make_interleaved_dataset(
        **dataset_kwargs,
        train=True,
    )

    dataset = (
        dataset.filter(has_language)
        .map(tokenizer.tokenize_language_instruction)
        .map(prepare_image)
    )

    if train:
        dataset = dataset.map(tokenizer.prepare_tokens_for_training)
    else:
        dataset = dataset.map(tokenizer.prepare_tokens_for_generation)

    return dataset


def has_language(data):
    return tf.strings.length(data["task"]["language_instruction"]) > 0


def component_label_fn(nested_params_dict):
    labels = {
        "llm": jax.tree.map(lambda _: "llm", nested_params_dict["llm"]),
        "img": jax.tree.map(lambda _: "img", nested_params_dict["img"]),
    }
    labels["llm"]["embedder"] = jax.tree.map(
        lambda _: "embed", nested_params_dict["llm"]["embedder"]
    )
    labels["img"]["head"] = jax.tree.map(
        lambda _: "embed", nested_params_dict["img"]["head"]
    )
    return labels


def make_optimizer(config: ConfigDict):
    @optax.inject_hyperparams
    def _make_optimizer(llm_learning_rate, img_learning_rate, embed_learning_rate):
        def _make_opt(lr, weight_decay, grad_norm_clip):
            if config.optimizer == "adamw":
                return optax.chain(
                    optax.clip_by_global_norm(grad_norm_clip),
                    optax.adamw(lr, weight_decay=weight_decay),
                )
            elif config.optimizer == "sgd":
                return optax.chain(
                    optax.clip_by_global_norm(grad_norm_clip),
                    optax.sgd(lr),
                )
            else:
                raise ValueError(f"Unknown optimizer: {config.optimizer}")

        img_optimizer = _make_opt(
            img_learning_rate,
            config.img_optimizer_kwargs.weight_decay,
            config.img_optimizer_kwargs.grad_norm_clip,
        )
        embed_optimizer = _make_opt(
            embed_learning_rate,
            config.embed_optimizer_kwargs.weight_decay,
            config.embed_optimizer_kwargs.grad_norm_clip,
        )
        llm_optimizer = _make_opt(
            llm_learning_rate,
            config.llm_optimizer_kwargs.weight_decay,
            config.llm_optimizer_kwargs.grad_norm_clip,
        )

        return optax.multi_transform(
            {
                "llm": llm_optimizer,
                "img": img_optimizer,
                "embed": embed_optimizer,
            },
            component_label_fn,
        )

    def _make_learning_rate(optimizer_kwargs):
        return optax.warmup_cosine_decay_schedule(
            optimizer_kwargs.init_learning_rate,
            optimizer_kwargs.learning_rate,
            optimizer_kwargs.warmup_steps,
            config.num_steps - optimizer_kwargs.warmup_steps,
        )

    return _make_optimizer(
        _make_learning_rate(config.llm_optimizer_kwargs),
        _make_learning_rate(config.img_optimizer_kwargs),
        _make_learning_rate(config.embed_optimizer_kwargs),
    )


def components_by_label(values):
    labels = component_label_fn(values)
    groups = {}
    for label, value in zip(jax.tree.leaves(labels), jax.tree.leaves(values)):
        groups[label] = value
    return groups


def smooth_nll_loss(logits, labels, sigma, base_action_token, action_vocab_size):
    # Normal PDF with mean `label` and std `sigma`
    bin_cutoffs = jnp.arange(action_vocab_size - 1) + 0.5
    cdf = jax.scipy.stats.norm.cdf(
        bin_cutoffs, labels[..., None] - base_action_token, sigma
    )
    cdf = jnp.concatenate(
        [jnp.zeros_like(cdf[..., :1]), cdf, jnp.ones_like(cdf[..., :1])], axis=-1
    )
    label_probs = jnp.diff(cdf, axis=-1)

    logits = jax.nn.log_softmax(logits, axis=-1)
    entropy = -jnp.sum(jax.scipy.special.xlogy(label_probs, label_probs), axis=-1)
    return (
        -jnp.sum(
            label_probs
            * logits[..., base_action_token : base_action_token + action_vocab_size],
            axis=-1,
        )
        - entropy
    )


def step_fn(
    train_state: TrainState,
    batch,
    key: chex.PRNGKey,
    tokenizer: Tokenizer.TokenizerConfig,
):
    def loss_fn(params, batch, key: chex.PRNGKey):
        batch_size, seq_len = batch["tokens"].shape

        chex.assert_shape(batch["image"], (batch_size, 224, 224, 3))
        chex.assert_shape(
            [batch["tokens"], batch["mask_ar"], batch["mask_loss"]],
            (batch_size, seq_len),
        )

        with jax.profiler.TraceAnnotation("apply_fn"):
            logits, out = train_state.apply_fn(
                {"params": params},
                batch["image"],
                batch["tokens"][..., :-1],
                # jnp.where(batch["tokens"][..., :-1] == tokenizer.pad_token, tokenizer.pad_token, tokenizer.begin_of_action_token),
                batch["mask_ar"][..., :-1],
                # train=True,
                # rngs={"dropout": key},
            )

        # from jax._src.debugging import inspect_sharding_p

        # def _visualize(k, v, sharding: jax.sharding.PositionalSharding):
        #     import time

        #     time.sleep(0.1)
        #     keystr = jax.tree_util.keystr(k)
        #     sharding_str = " ".join(repr(sharding).split())
        #     try:
        #         print(f"Visualizing {keystr} ({v.shape}): {sharding_str}")
        #         jax.debug.visualize_sharding(v.shape, sharding)
        #     except:
        #         print(f"Visualizing {keystr} ({jax.numpy.shape(v)}): {sharding_str}")
        #     time.sleep(0.1)

        # def _inspect(path, val):
        #     inspect_sharding_p.bind(val, callback=lambda x: _visualize(path, val, x))

        # jax.tree_util.tree_map_with_path(
        #     _inspect,
        #     {"data": batch, "params": train_state.params, "intermediates": out},
        # )

        output_pred_mask = batch["mask_loss"][..., 1:]
        labels = batch["tokens"][..., 1:]

        chex.assert_shape([logits], (batch_size, seq_len - 1, tokenizer.vocab_size))

        loss_by_token = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        # loss_by_token = smooth_nll_loss(
        #     logits,
        #     labels,
        #     sigma=5,
        #     base_action_token=tokenizer.action_vocab_offset,
        #     action_vocab_size=tokenizer.action_vocab_size,
        # )
        chex.assert_shape(
            [labels, output_pred_mask, loss_by_token], (batch_size, seq_len - 1)
        )

        loss = jnp.mean(loss_by_token * output_pred_mask) / jnp.mean(output_pred_mask)

        pred_tokens = jnp.argmax(logits, axis=-1)
        accuracy_by_token = pred_tokens == labels
        accuracy = jnp.mean(output_pred_mask * accuracy_by_token) / jnp.mean(
            output_pred_mask
        )

        chex.assert_equal_shape([loss_by_token, output_pred_mask, accuracy_by_token])

        # Decode actions
        get_action_tokens = jax.vmap(
            lambda x, i: jax.lax.dynamic_slice(x, (i,), (tokenizer.num_action_tokens,))
        )

        action_token_starts = (
            jnp.argmax(labels == tokenizer.begin_of_action_token, axis=-1) + 1
        )
        pred_action_tokens = get_action_tokens(pred_tokens, action_token_starts)
        gt_action_tokens = get_action_tokens(labels, action_token_starts)
        loss_by_action_token = get_action_tokens(loss_by_token, action_token_starts)
        accuracy_by_action_token = get_action_tokens(
            accuracy_by_token, action_token_starts
        )

        decoded_actions = tokenizer.bin_detokenize(pred_action_tokens)
        decoded_actions_gt = tokenizer.bin_detokenize(gt_action_tokens)
        mae = jnp.abs(decoded_actions - decoded_actions_gt)
        mse = jnp.square(decoded_actions - decoded_actions_gt)

        details = {}
        for i in range(tokenizer.num_action_tokens):
            details[f"details/tf_loss_{i}"] = jnp.mean(loss_by_action_token[:, i])
            details[f"details/tf_accuracy_{i}"] = jnp.mean(
                accuracy_by_action_token[:, i]
            )

        for i in range(decoded_actions.shape[1]):
            details[f"details/tf_mse_{i}"] = jnp.mean(mse[:, i])
            details[f"details/tf_mae_{i}"] = jnp.mean(mae[:, i])

        return (
            loss,
            {
                "loss": loss,
                "accuracy": accuracy,
                "tf_l1": jnp.mean(mae),
                "tf_l2": jnp.mean(mse),
                "norm/img_embed": optax.global_norm(out["img/zimg"]),
                "norm/llm_embed": optax.global_norm(out["llm/ztxt"]),
                # "img_embed_norm": out["img/embed_norm"],
                # "llm_embed_norm": out["llm/embed_norm"],
                # "img_out_norm": out["img/out_norm"],
                # "llm_out_norm": out["llm/out_norm"],
            }
            | details,
        )

    with jax.profiler.TraceAnnotation("grad"):
        grad_fn = jax.grad(loss_fn, has_aux=True)

    with jax.profiler.TraceAnnotation("apply_updates"):
        key, dropout_key = jax.random.split(key)
        grads, info = grad_fn(train_state.params, batch, dropout_key)
        updates, opt_state = train_state.tx.update(
            grads, train_state.opt_state, params=train_state.params
        )
        params = optax.apply_updates(train_state.params, updates)

        train_state = train_state.replace(
            params=params, opt_state=opt_state, step=train_state.step + 1
        )

    info = info | train_state.opt_state.hyperparams

    def _norm_info(values, prefix):
        components = components_by_label(values)
        result = {f"{prefix}_{k}": optax.global_norm(v) for k, v in components.items()}
        result[prefix] = jnp.sqrt(sum(x**2 for x in result.values()))
        return result

    info = (
        info
        | _norm_info(grads, "norm/grad")
        | _norm_info(updates, "norm/update")
        | _norm_info(train_state.params, "norm/param")
    )

    return train_state, info, key


def host_broadcast_str(x: str | None) -> str:
    """Broadcast_one_to_all, but with a string. Strings should all be the same length."""
    if x is None:
        x = ""

    max_len = multihost_utils.broadcast_one_to_all(len(x))
    padded = x.ljust(max_len)

    encoded = np.array([ord(c) for c in padded], dtype=np.uint8)[:max_len]
    encoded = multihost_utils.broadcast_one_to_all(encoded)
    decoded = "".join([chr(u) for u in encoded])

    return decoded.rstrip()


def main(_):
    jax.distributed.initialize()

    jax_smi.initialise_tracking()
    tf.random.set_seed(jax.process_index())

    config = flags.FLAGS.config

    with open(config.tokenizer_path, "rb") as f:
        language_tokenizer = SentencepieceTokenizer(f.read())
    tokenizer = Tokenizer.from_tokenizer(language_tokenizer)

    print("Constructing dataset...")
    ds_train = make_dataset(config, tokenizer, train=True)
    ds_eval = make_dataset(config, tokenizer, train=False)

    train_it = ds_train.batch(config.batch_size // jax.process_count()).iterator()
    eval_it = ds_eval.batch(config.eval_batch_size // jax.process_count()).iterator()

    print("Loading model params...")
    model, params, decode = load_model(config, tokenizer)

    print("Initializing model...")

    optimizer = make_optimizer(config)

    mesh = MeshShardingHelper([config.data_axis_size, config.fsdp_axis_size], ["data", "fsdp"])  # , mesh_axis_splitting=True)

    model_sharding = FSDPShardingRule("fsdp", fsdp_axis_size=mesh.mesh.shape["fsdp"])
    data_sharding = PartitionSpec(("data", "fsdp"))

    @partial(mesh.sjit, in_shardings=None, out_shardings=model_sharding)
    def init_fn(params):
        return TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optimizer,
        )

    train_state = init_fn(params)
    del params

    key = jax.random.PRNGKey(0)

    if jax.process_index() == 0:
        if config.profile:
            wandb.init(project="paligemma-vla", mode="disabled")
        else:
            wandb.init(project="paligemma-vla")
            wandb.config.update(config.to_dict())

        run_name = wandb.run.name
    else:
        run_name = None

    run_name = host_broadcast_str(run_name)

    if config.save_path is not None:
        checkpoint_manager = ocp.CheckpointManager(
            f"{config.save_path}/{run_name}",
            ocp.PyTreeCheckpointer(),
            options=ocp.CheckpointManagerOptions(max_to_keep=3),
        )


    jit_step_fn = mesh.sjit(
        step_fn,
        in_shardings=(model_sharding, data_sharding, None),
        out_shardings=(model_sharding, None, None),
        static_argnums=(3,),
        args_sharding_constraint=(
            model_sharding,
            data_sharding,
            None,
        ),
        donate_argnums=(0,),
    )
    jit_step_fn = jax.profiler.annotate_function(jit_step_fn, name="step_fn")

    if config.profile:
        config.num_steps = 10

    wandb_logs = []

    tracemalloc.start()
    with tqdm.trange(config.num_steps, desc="Training") as pbar:
        for i in pbar:
            if i == 2 and jax.process_index() == 0:
                jax.profiler.start_trace("/tmp/jax-trace")

            batch = next(train_it)
            batch_train = mesh.local_data_to_global_array(
                {
                    "image": np.squeeze(batch["observation"]["image_primary"], axis=1),
                    "tokens": batch["tokens"],
                    "input_mask": batch["mask_input"],
                    "mask_ar": batch["mask_ar"],
                    "mask_loss": batch["mask_loss"],
                }
            )

            with jax.profiler.StepTraceAnnotation("step_fn", step_num=i):
                from flax import linen as nn
                with mesh.mesh, nn.logical_axis_rules([("act_batch", "fsdp")]):
                    train_state, info, key = jit_step_fn(
                        train_state,
                        batch_train,
                        key,
                        tokenizer.config,
                    )
                train_state = jax.block_until_ready(train_state)

            pbar.set_postfix(
                loss=f"{info['loss']:.4f}",
            )
            wandb_logs.append(jax.device_get(info))

            if (i + 1) % config.log_interval == 0:
                avg_info = jax.tree.map(
                    lambda *xs: jnp.mean(jnp.stack(xs), axis=0), *wandb_logs
                )
                avg_info = jax.device_get(avg_info)
                if jax.process_index() == 0:
                    wandb.log(avg_info, step=i)
                wandb_logs = []

            if (i + 1) % 100 == 0:
                # Print out a trace of the largest objects in memory
                def display_top(snapshot, key_type='lineno', limit=10):
                    snapshot = snapshot.filter_traces((
                        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
                        tracemalloc.Filter(False, "<unknown>"),
                    ))
                    top_stats = snapshot.statistics(key_type)
                    
                    print(f"Top {limit} lines")
                    for index, stat in enumerate(top_stats[:limit], 1):
                        frame = stat.traceback[0]
                        print(f"#{index}: {frame.filename}:{frame.lineno}: {stat.size/1024/1024:.1f} MiB")
                        line = linecache.getline(frame.filename, frame.lineno).strip()
                        if line:
                            print(f"    {line}")
                    
                    other = top_stats[limit:]
                    if other:
                        size = sum(stat.size for stat in other)
                        print(f"{len(other)} other: {size/1024/1024:.1f} MiB")
                    total = sum(stat.size for stat in top_stats)
                    print(f"Total allocated size: {total/1024/1024:.1f} MiB")

                snapshot = tracemalloc.take_snapshot()
                display_top(snapshot)

            if (i + 1) % config.eval_interval == 0:
                batch_eval = next(eval_it)
                batch_eval = mesh.local_data_to_global_array(
                    {
                        "image": batch_eval["observation"]["image_primary"][:, 0],
                        "text": batch_eval["tokens"],
                        "mask_ar": batch_eval["mask_ar"],
                        "mask_input": batch_eval["mask_input"],
                        "action": np.squeeze(batch_eval["action"], axis=(1, 2)),
                        "_mask": np.ones(batch_eval["tokens"].shape[0], dtype=np.bool_),
                    }
                )
                out_tokens = decode(
                    {"params": train_state.params},
                    batch_eval,
                    model=model,
                    devices=jax.devices(),
                    max_decode_len=tokenizer.config.num_action_tokens,
                    replicate_out=True,
                    mesh=mesh.mesh,
                )
                out_tokens = jax.device_get(
                    multihost_utils.process_allgather(out_tokens)
                )

                decoded_actions = tokenizer.bin_detokenize(out_tokens)

                gt_action = jax.device_get(
                    multihost_utils.process_allgather(batch_eval["action"])
                )

                info = {
                    "rollout_mae": np.mean(np.abs(decoded_actions - gt_action)),
                    "rollout_mse": np.mean(np.square(decoded_actions - gt_action)),
                }
                if jax.process_index() == 0:
                    wandb.log(
                        info,
                        commit=False,
                        step=i,
                    )

            if (i + 1) % config.save_interval == 0:
                if config.save_path is not None:
                    print(f"Saving model to {config.save_path}/{i}")
                    checkpoint_manager.save(i + 1, train_state)

    if config.profile and jax.process_index() == 0:
        jax.profiler.stop_trace()


if __name__ == "__main__":
    config_flags.DEFINE_config_file(
        "config", "bridge_config.py", "Path to the config file."
    )
    app.run(main)
