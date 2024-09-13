import tensorflow as tf
from ml_collections import ConfigDict

from palivla.tokenizer import Tokenizer
from octo.data.dataset import make_interleaved_dataset
from octo.data.oxe import make_oxe_dataset_kwargs_and_weights


def prepare_image(data):
    data["observation"]["image_primary"] = (
        tf.cast(data["observation"]["image_primary"], tf.float32) / 127.5 - 1
    )
    return data


def make_dataset(config: ConfigDict, tokenizer: Tokenizer, train: bool, generation: bool):
    dataset_kwargs = config.dataset_kwargs.to_dict()

    dataset_kwargs["dataset_kwargs_list"], dataset_kwargs["sample_weights"] = (
        make_oxe_dataset_kwargs_and_weights(**dataset_kwargs.pop("oxe_kwargs"))
    )

    dataset = make_interleaved_dataset(
        **dataset_kwargs,
        train=train,
    )
    dataset_statistics = dataset.dataset_statistics

    dataset = (
        dataset.filter(has_language)
        .map(tokenizer.tokenize_language_instruction, num_parallel_calls=None)
        .map(prepare_image, num_parallel_calls=None)
    )

    if generation:
        dataset = dataset.map(
            tokenizer.prepare_tokens_for_generation, num_parallel_calls=None
        )
    else:
        dataset = dataset.map(
            tokenizer.prepare_tokens_for_training, num_parallel_calls=None
        )

    dataset.dataset_statistics = dataset_statistics

    return dataset


def has_language(data):
    return tf.strings.length(data["task"]["language_instruction"]) > 0
