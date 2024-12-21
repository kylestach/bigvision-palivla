import optax
import jax
import fnmatch

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


def make_optimizer(**config):
    @optax.inject_hyperparams
    def _make_optimizer(llm_learning_rate, img_learning_rate, embed_learning_rate):
        def _make_opt(lr, weight_decay, grad_norm_clip, b1, b2):
            if config["optimizer"] == "adamw":
                return optax.chain(
                    optax.clip_by_global_norm(grad_norm_clip),
                    optax.adamw(lr, weight_decay=weight_decay, b1=b1, b2=b2),
                )
            elif config["optimizer"] == "sgd":
                return optax.chain(
                    optax.clip_by_global_norm(grad_norm_clip),
                    optax.sgd(lr),
                )
            else:
                raise ValueError(f"Unknown optimizer: {config['optimizer']}")

        img_optimizer = _make_opt(
            img_learning_rate,
            config["img_optimizer_kwargs"]["weight_decay"],
            config["img_optimizer_kwargs"]["grad_norm_clip"],
            config["img_optimizer_kwargs"].get("b1", 0.9),
            config["img_optimizer_kwargs"].get("b2", 0.999),
        )
        embed_optimizer = _make_opt(
            embed_learning_rate,
            config["embed_optimizer_kwargs"]["weight_decay"],
            config["embed_optimizer_kwargs"]["grad_norm_clip"],
            config["embed_optimizer_kwargs"].get("b1", 0.9),
            config["embed_optimizer_kwargs"].get("b2", 0.999),
        )
        llm_optimizer = _make_opt(
            llm_learning_rate,
            config["llm_optimizer_kwargs"]["weight_decay"],
            config["llm_optimizer_kwargs"]["grad_norm_clip"],
            config["llm_optimizer_kwargs"].get("b1", 0.9),
            config["llm_optimizer_kwargs"].get("b2", 0.999),
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
            optimizer_kwargs["init_learning_rate"],
            optimizer_kwargs["learning_rate"],
            optimizer_kwargs["warmup_steps"],
            config["num_steps"] - optimizer_kwargs["warmup_steps"],
        )

    return _make_optimizer(
        _make_learning_rate(config["llm_optimizer_kwargs"]),
        _make_learning_rate(config["img_optimizer_kwargs"]),
        _make_learning_rate(config["embed_optimizer_kwargs"]),
    )
