import os
import time 

from big_vision.utils import Registry
from palivla.components.action_tokenizer import ActionTokenizer
from palivla.components.model import PaliVLAModel
from palivla.components.sequence_builder import SequenceBuilder
from palivla.components.train_state import ShardingMetadata

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
print("CUDA VISIBLE DEVICES: ", os.environ["CUDA_VISIBLE_DEVICES"])


import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import tensorflow as tf
import tqdm
from absl import app, flags
from absl import logging as absl_logging
from flax.core.frozen_dict import freeze
from ml_collections import ConfigDict, config_flags
from scalax.sharding import FSDPShardingRule, MeshShardingHelper
from transformers import AutoTokenizer

physical_devices = tf.config.list_physical_devices('GPU')
print("TF VISIBLE DEVICES", physical_devices)
tf.config.set_visible_devices(physical_devices, "GPU")
print("JAX VISIBLE DEVICES: ", jax.devices())

import wandb
import palivla.load_fns
from palivla.dataset import make_base_dataset
from palivla.model_components import ModelComponents
from palivla.optimizer import make_optimizer
from palivla.spec import ModuleSpec, OptimizerSpec
from palivla.utils import host_broadcast_str

from palivla.frodo_dataset import FrodoDataset
from palivla.train_step import TrainingBatch

import torch 
from torch.utils.data import DataLoader

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
tf.config.set_visible_devices([], "GPU")

os.environ["OMP_NUM_THREADS"] = "12"  # Set number of OpenMP threads
os.environ["MKL_NUM_THREADS"] = "12"  # Set number of MKL threads
torch.set_num_threads(12)  # Limit the number of CPU threads used by PyTorch (edited) 


import multiprocessing as mp
from functools import partial


def get_item(j, dataset):
    return dataset[j]


def collate_fn(batch):

    # def check_shapes(key, tensors):
    #     shapes = [x.shape for x in tensors]
    #     first_shape = shapes[0]
    #     for i, shape in enumerate(shapes):
    #         if shape != first_shape:
    #             print(f"Shape mismatch in '{key}' at batch index {i}: got {shape}, expected {first_shape}")
    #     return shapes

    # # Check observation shapes
    # for key in ["image_front", "image_goal", "gps_goal"]:
    #     check_shapes(f"observation/{key}", [b["observation"][key] for b in batch])

    # # Check action shapes
    # check_shapes("action", [b["action"] for b in batch])

    # # Check pad_mask_dict shapes
    # for key in ["image_front", "image_goal", "gps_goal"]:
    #     check_shapes(f"observation/pad_mask_dict/{key}", [
    #         b["observation"]["pad_mask_dict"][key] for b in batch
    #     ])
    # check_shapes("pad_mask_dict/action", [b["pad_mask_dict"]["action"] for b in batch])

    return {
        "observation": {
            "image_front": np.stack(
                [b["observation"]["image_front"] for b in batch]
            ),
            "image_goal": np.stack(
                [b["observation"]["image_goal"] for b in batch]
            ),
            "gps_goal": np.stack(
                [b["observation"]["gps_goal"] for b in batch]
            ),
            "pad_mask_dict": {
                k: np.stack(
                    [
                        b["observation"]["pad_mask_dict"][k]
                        for b in batch
                    ]
                )
                for k in ["image_front", "image_goal", "gps_goal"]
            },
        },
        "action": np.stack([b["action"] for b in batch]),
        "pad_mask_dict": {
            "action": np.stack([b["pad_mask_dict"]["action"] for b in batch]),
        },
        "task" : {
            "language_instruction": [b["task"]["language_instruction"] for b in batch]
        }
    }
    



def make_sharding(config: ConfigDict):
    mesh = MeshShardingHelper([-1], ["fsdp"])
    sharding_metadata = ShardingMetadata(
        mesh=mesh,
        model_sharding_rule=FSDPShardingRule(
            "fsdp", fsdp_axis_size=mesh.mesh.shape["fsdp"]
        ),
    )
    return sharding_metadata


def create_model(config: ConfigDict, sharding_metadata: ShardingMetadata):
    example_batch = {
        "sensors": {
            "image_front": jax.ShapeDtypeStruct(
                shape=(1, 224, 224, 3), dtype=jnp.uint8
            ),
            "image_goal": jax.ShapeDtypeStruct(
                shape=(1, 224, 224, 3), dtype=jnp.uint8
            ),
            # "proprio": jax.ShapeDtypeStruct(shape=(1, 7), dtype=jnp.float32),
        },
        "sensors_mask": {
            "image_front": jax.ShapeDtypeStruct(
                shape=(1, 224, 224, 3), dtype=jnp.bool_
            ),
            "image_goal": jax.ShapeDtypeStruct(
                shape=(1, 224, 224, 3), dtype=jnp.bool_
            ),
            # "proprio": jax.ShapeDtypeStruct(shape=(1, 7), dtype=jnp.bool_),
        },
        "prompt": {
            "tokens": jax.ShapeDtypeStruct(shape=(1, 10), dtype=jnp.int32),
            "mask": jax.ShapeDtypeStruct(shape=(1, 10), dtype=jnp.bool_),
            "mask_ar": jax.ShapeDtypeStruct(shape=(1, 10), dtype=jnp.bool_),
            "mask_loss": jax.ShapeDtypeStruct(shape=(1, 10), dtype=jnp.float32),
        },
        "gen": {
            "tokens": jax.ShapeDtypeStruct(shape=(1, 10), dtype=jnp.int32),
            "mask": jax.ShapeDtypeStruct(shape=(1, 10), dtype=jnp.bool_),
            "mask_ar": jax.ShapeDtypeStruct(shape=(1, 10), dtype=jnp.bool_),
            "mask_loss": jax.ShapeDtypeStruct(shape=(1, 10), dtype=jnp.float32),
        },
    }

    language_tokenizer = AutoTokenizer.from_pretrained(config.language_tokenizer)
    action_tokenizer: ActionTokenizer = Registry.lookup(config.action_tokenizer)()
    sequence_builder: SequenceBuilder = Registry.lookup(config.sequence_builder)()

    extra_tokens = [
        "<begin_of_action>",
    ] + [f"<act{i}>" for i in range(action_tokenizer.vocab_size)]
    language_tokenizer.add_tokens(extra_tokens)
    language_tokenizer.add_bos_token = False

    model_config = config.model_config.to_dict()
    model_config["llm_spec"]["config"]["vocab_size"] = len(language_tokenizer)
    model_spec = ModuleSpec(
        PaliVLAModel,
        freeze(model_config),
    )
    optimizer_spec = OptimizerSpec.create(
        make_optimizer,
        config.optimizer.kwargs.to_dict(),
    )

    return ModelComponents.initialize(
        model_spec=model_spec,
        optimizer_spec=optimizer_spec,
        seed=config.get("seed", 0),
        language_tokenizer=language_tokenizer,
        action_tokenizer=action_tokenizer,
        sequence_builder=sequence_builder,
        sharding_metadata=sharding_metadata,
        example_batch=(example_batch["sensors"], example_batch["sensors_mask"], example_batch["prompt"], example_batch["gen"]),
    )


def main(_):
    if flags.FLAGS.platform == "tpu":
        jax.distributed.initialize()

    # Turn off debug logs
    tf.get_logger().setLevel("WARNING")
    absl_logging.set_verbosity(absl_logging.WARNING)

    tf.random.set_seed(jax.process_index())

    config = flags.FLAGS.config

    sharding_metadata = make_sharding(config)
    print("Sharding set up")

    if config.resume_checkpoint_dir is not None:
        # Load the model from a checkpoint
        model = ModelComponents.load_static(
            config.resume_checkpoint_dir, sharding_metadata
        )
        restore_manager = ocp.CheckpointManager(
            config.resume_checkpoint_dir, options=ocp.CheckpointManagerOptions()
        )
        model.load_state(config.resume_checkpoint_step, restore_manager)
    else:
        # Otherwise, create the model from scratch and apply any load_fns
        model = create_model(config, sharding_metadata)

        print("model sharding mesh", model.sharding.mesh)
        for load_fn, load_fn_kwargs in config.load_fns:
            print("Load function", load_fn, "and kwargs", load_fn_kwargs)
            load_fn = Registry.lookup(load_fn)
            load_fn(model, **load_fn_kwargs)

    print("Model set up")
    # Make the basic dataset
    # We have to do this first, since we need to know how the dataset is set up before we can construct the model
    # train_ds = make_base_dataset(**config.dataset_kwargs.to_dict(), train=True)

    start_time = time.time()
    train_ds = FrodoDataset(
        config.dataset,
        root=config.dataset_root,
        split="train",
        action_horizon=8,
        action_spacing=5,
        goal_horizon=100,
        context_size=0,
        image_size=(224, 224),
        load_goal_image=True,
        action_key = "action_mbra",
    )
    print(f"Dataset loaded in {time.time() -start_time}")

    per_host_train_batch_size = config.batch_size // jax.process_count()
    per_host_eval_batch_size = config.eval_batch_size // jax.process_count()

    mesh = MeshShardingHelper([-1], ["fsdp"])

    def make_training_batch(batch):
        return batch
        # sensors = {
        #     k: batch["observation"][k]
        #     for k in batch["observation"]
        #     if k in model.model_state.model.modality_mappings and k != "text"
        # }
        # sensors_mask = {
        #     k: np.squeeze(batch["observation"]["pad_mask_dict"][k], axis=-1)
        #     for k in model.model_state.model.modality_mappings
        #     if k != "text"
        # }

        # tokens = jax.device_get(model.tokenize_action(batch["action"], None))
        # tokens = np.concatenate(
        #     [np.zeros((tokens.shape[0], 1), dtype=tokens.dtype), tokens], axis=-1
        # )
        # tokens_ar = np.ones_like(tokens)
        # tokens_mask = np.ones_like(tokens)
        # tokens_loss = np.ones_like(tokens)

        # # Start of generation
        # gen_start = np.ones((tokens.shape[0],), dtype=np.int32)

        # # print("Tokens")
        # # print(np.where(tokens_loss, tokens, -1)[0])
        # # print(tokens[0])
        # # breakpoint()


        # return mesh.local_data_to_global_array(
        #     TrainingBatch(
        #         sensors=sensors,
        #         sensors_mask=sensors_mask,
        #         actions=batch["action"],
        #         actions_mask=batch["pad_mask_dict"]["action"],
        #         tokens=tokens,
        #         tokens_ar=tokens_ar,
        #         tokens_loss=tokens_loss,
        #         tokens_mask=tokens_mask,
        #         gen_start=gen_start,
        #     )
        # )
    
    # train_ds.item_transform = make_frame_transform(generation=False, tokenizer=model.tokenizer)
    # frame_transform = make_frame_transform(generation=False, tokenizer=model.tokenizer)

    def _collate_fn(batch):
        b0 = batch[0]
        if isinstance(b0, dict):
            return {k: _collate_fn([b[k] for b in batch]) for k in b0}
        elif isinstance(b0, (np.ndarray, torch.Tensor)):
            return np.stack(batch)
        elif isinstance(b0, str):
            return None

        raise ValueError(f"Unknown batch type: {type(batch)}")

    # breakpoint()
    # train_loader = DataLoader(
    #     train_ds,
    #     batch_size=config.batch_size,
    #     # sampler=train_ds.get_sampler(),
    #     shuffle = True,
    #     num_workers = 0,
    #     # num_workers=128,
    #     collate_fn=collate_fn,
    #     # multiprocessing_context='forkserver', # don't fork - jax gets mad 
    # )
    # breakpoint()
    
    # train_it = map(make_training_batch, iter(train_loader))



    print("Dataset iterator set up")


    # Construct the final dataset
    # We need to do this after the model is constructed, since we need to have a tokenizer
    # per_host_train_batch_size = config.batch_size // jax.process_count()

    # def make_training_batch(batch):
    #     return batch

    # train_it = map(
    #     make_training_batch,
    #     train_ds.batch(per_host_train_batch_size).iterator(),
    # )

    # W&B setup
    if jax.process_index() == 0:
        wandb_kwargs = {
            "project": config.wandb_project,
            "tags": [],
            "mode": config.wandb_mode,
        }

        wandb.init(**wandb_kwargs)
        wandb.config.update(config.to_dict())

        run_name = wandb.run.name
    else:
        run_name = None

    run_name = host_broadcast_str(run_name)

    if config.save_path is not None:
        checkpoint_save_path = tf.io.gfile.join(config.save_path, run_name)

        checkpoint_save_manager = ocp.CheckpointManager(
            checkpoint_save_path,
            options=ocp.CheckpointManagerOptions(max_to_keep=config.max_to_keep),
        )

    wandb_logs = []

    # Main training loop
    start_step = model.train_state.step.item()

    # if config.overfit_dataset:
    #     batch = next(train_it)

    # with tqdm.trange(
    #     start_step, config.num_steps, desc="Training", dynamic_ncols=True
    # ) as pbar:
    #     for i in pbar:

    # num_batches = len(train_loader)
    # print("num batches is", num_batches)

    # breakpoint()

    # tqdm_iter = tqdm.tqdm(
    #     train_loader,
    #     disable = False,
    #     dynamic_ncols = True,
    #     desc= f"Progress through dataset"
    # )
    # for i, batch in enumerate(tqdm_iter):

    idxs = np.arange(1, len(train_ds))  # from 1 to 1233093 inclusive
    np.random.shuffle(idxs)
    print("idxs set up")


    # partial_get_item = partial(get_item, dataset=train_ds)

    for i in tqdm.tqdm(range(len(train_ds) // config.batch_size)): 

        # if not config.overfit_dataset:
            # batch = next(train_it)


        # with mp.get_context("forkserver").Pool(processes=32) as pool:  # or whatever number of processes makes sense
        #     batch = list(tqdm.tqdm(pool.imap(partial_get_item, idxs[i * config.batch_size:(i + 1) * config.batch_size]), 
        #                     total=config.batch_size, 
        #                     desc="Loading batch"))


        batch = [train_ds[j] for j in tqdm.tqdm(idxs[i * config.batch_size:(i + 1) * config.batch_size], desc="Loading batch")]

        # batch = [train_ds[j] for j in idxs[i* config.batch_size:(i+1)* config.batch_size]]
        batch = collate_fn(batch)
        batch["action"] = batch["action"][:, np.newaxis, :, :] # to work with sequence generator 

        info = model.train_step(batch)
        info = jax.device_get(info)
        wandb_logs.append(info)
        # pbar.set_postfix(
        #     loss=f"{info['loss']:.4f}",
        # )

        if (i + 1) % config.eval_interval == 0:
            eval_info = model.eval_step(batch)
            if jax.process_index() == 0:
                wandb.log(eval_info, step=i + 1, commit=False)

        if (i + 1) % config.log_interval == 0:
            avg_info = jax.tree.map(
                lambda *xs: np.mean(np.stack(xs), axis=0), *wandb_logs
            )
            if jax.process_index() == 0:
                wandb.log(avg_info, step=i + 1)
            wandb_logs = []

        if (i + 1) % config.save_interval == 0:
            if config.save_path is not None:
                checkpoint_save_manager.save(i + 1, args=model.save_args())

    if config.save_path is not None:
        checkpoint_save_manager.wait_until_finished()


if __name__ == "__main__":
    config_flags.DEFINE_config_file(
        "config", "configs/smoke_test.py", "Path to the config file."
    )
    flags.DEFINE_string("platform", "gpu", "Platform to run on.")
    app.run(main)
