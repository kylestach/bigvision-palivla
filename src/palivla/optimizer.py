import fnmatch

import jax
import optax

from big_vision.utils import Registry
from palivla.utils import key_string


def component_label_fn(nested_params_dict):
    label_rules = [
        # Assign the input/output embeddings and the image connector to "embed"
        ("*/embedder", "embed"),
        ("img/head", "embed"),
        # Assign the llm and the image encoder to "llm" and "img" respectively
        ("llm/*", "llm"),
        ("img/*", "img"),
        # Assign the rest to "embed"
        # This includes any other modality-specific encoders
        ("*", "embed"),
    ]

    def label_fn(path, _):
        path_str = key_string(path)
        for pattern, label in label_rules:
            if fnmatch.fnmatch(path_str, pattern):
                return label
        return path_str

    return jax.tree_util.tree_map_with_path(label_fn, nested_params_dict)


def components_by_label(values):
    labels = component_label_fn(values)
    groups = {}
    for label, value in zip(jax.tree.leaves(labels), jax.tree.leaves(values)):
        groups[label] = value
    return groups


def ema_params(rate: float):
    def _init_fn(params):
        return params

    def _update_fn(updates, state, params):
        return updates, optax.incremental_update(state, params, rate)

    return optax.GradientTransformation(
        init=_init_fn,
        update=_update_fn,
    )


@Registry.register("optimizer.default_optimizer")
def make_optimizer(
    optimizer: str,
    num_train_steps: int,
    base_learning_rate: float = 1e-4,
    img_optimizer_kwargs: dict = {},
    embed_optimizer_kwargs: dict = {},
    llm_optimizer_kwargs: dict = {},
    ema_rate: float | None = None,
):
    @optax.inject_hyperparams
    def _make_optimizer(llm_learning_rate, img_learning_rate, embed_learning_rate):
        def _make_opt(
            lr, weight_decay=1e-4, grad_norm_clip=10.0, b1=0.9, b2=0.999, **kwargs
        ):
            if optimizer == "adamw":
                return optax.chain(
                    optax.clip_by_global_norm(grad_norm_clip),
                    optax.adamw(lr, weight_decay=weight_decay, b1=b1, b2=b2),
                )
            elif optimizer == "sgd":
                return optax.chain(
                    optax.clip_by_global_norm(grad_norm_clip),
                    optax.sgd(lr),
                )
            else:
                raise ValueError(f"Unknown optimizer: {optimizer}")

        img_optimizer = _make_opt(
            img_learning_rate,
            **img_optimizer_kwargs,
        )
        embed_optimizer = _make_opt(
            embed_learning_rate,
            **embed_optimizer_kwargs,
        )
        llm_optimizer = _make_opt(
            llm_learning_rate,
            **llm_optimizer_kwargs,
        )

        return optax.multi_transform(
            {
                "llm": llm_optimizer,
                "img": img_optimizer,
                "embed": embed_optimizer,
            },
            component_label_fn,
        )

    def _make_learning_rate(
        learning_rate=base_learning_rate,
        init_learning_rate=0.0,
        warmup_steps=1000,
        **kwargs,
    ):
        return optax.warmup_cosine_decay_schedule(
            init_learning_rate,
            learning_rate,
            warmup_steps,
            num_train_steps - warmup_steps,
        )

    transforms = [
        (
            "optimizer",
            _make_optimizer(
                _make_learning_rate(**llm_optimizer_kwargs),
                _make_learning_rate(**img_optimizer_kwargs),
                _make_learning_rate(**embed_optimizer_kwargs),
            ),
        ),
    ]

    if ema_rate is not None:
        transforms.append(("ema", ema_params(ema_rate)))

    return optax.named_chain(*transforms)
