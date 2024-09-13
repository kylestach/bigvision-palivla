import jax
import jax.numpy as jnp
import chex
import optax
from flax.training.train_state import TrainState
from palivla.tokenizer import Tokenizer
from palivla.model import components_by_label

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
                batch["mask_ar"][..., :-1],
                # train=True,
                # rngs={"dropout": key},
            )


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

