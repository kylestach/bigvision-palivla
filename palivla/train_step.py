from functools import partial
from typing import Dict
import jax
import jax.numpy as jnp
import chex
import optax
from flax.training.train_state import TrainState
from palivla.tokenizer import ActionTokenizer, Tokenizer
from palivla.load_model import components_by_label
from palivla.types import TrainingBatch
from big_vision.distributional import cross_entropy_loss_on_scalar, hl_gauss_transform
from palivla.types import RolloutBatch


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


def get_action_tokens(
    pred_logits, tokens, num_action_tokens: int, begin_of_action_token: int
):
    _get_action_tokens = jax.vmap(
        lambda x, i: jax.lax.dynamic_slice(x, (i,), (num_action_tokens,))
    )

    action_token_starts = jnp.argmax(tokens == begin_of_action_token, axis=-1) + 1
    pred_tokens = jnp.argmax(pred_logits, axis=-1)
    pred_action_tokens = _get_action_tokens(pred_tokens, action_token_starts)
    pred_action_logits = jax.vmap(_get_action_tokens, in_axes=(-1, None), out_axes=-1)(
        pred_logits, action_token_starts
    )
    gt_action_tokens = _get_action_tokens(tokens, action_token_starts) # gt = ground truth

    return {
        "pred_action_tokens": pred_action_tokens,
        "pred_action_logits": pred_action_logits,
        "gt_action_tokens": gt_action_tokens,
    }

def replace_action_tokens(
    tokens, target_action_tokens, begin_of_action_token: int
):
    _replace_action_tokens = jax.vmap(
        lambda x, i, s: jax.lax.dynamic_update_slice(x, i, (s,))
    )
    action_token_starts = jnp.argmax(tokens == begin_of_action_token, axis=-1) + 1
    return _replace_action_tokens(tokens, target_action_tokens, action_token_starts)
    

def compute_action_metrics(
    detokenize_fn,
    *,
    pred_action_tokens,
    pred_action_logits,
    gt_action_tokens,
    gt_actions,
    action_dim: int,
    tokenizer_config: Tokenizer.TokenizerConfig,
    log_segment_prefix=None,
):  
    decoded_actions = detokenize_fn(pred_action_tokens)
    decoded_actions_gt = detokenize_fn(gt_action_tokens)

    batch_size = gt_actions.shape[0]
    chex.assert_shape((pred_action_tokens, gt_action_tokens), (batch_size, tokenizer_config.num_action_tokens))
    if pred_action_logits is not None:
        chex.assert_shape((pred_action_logits), (batch_size, tokenizer_config.num_action_tokens, None))
    chex.assert_shape((gt_actions, decoded_actions_gt, decoded_actions), (batch_size, None, action_dim))

    def stats_for_metric(value, name, shape):
        chex.assert_shape(value, (None, shape))
        return {
            f"{log_segment_prefix}{name}": jnp.mean(value),
        } | {
            f"_details/{log_segment_prefix}{name}_{i}": jnp.mean(value[:, i])
            for i in range(shape)
        }

    error = decoded_actions - decoded_actions_gt
    tokenization_error = gt_actions - decoded_actions_gt

    if tokenizer_config.min_action_value is not None:
        tokenization_error = jnp.clip(
            gt_actions,
            tokenizer_config.min_action_value,
            tokenizer_config.max_action_value
        ) - decoded_actions_gt

    return {
        **stats_for_metric(jnp.mean(jnp.abs(error), axis=1), "l1", action_dim),
        **stats_for_metric(jnp.mean(jnp.square(error), axis=1), "l2", action_dim),
        **stats_for_metric(
            jnp.mean(jnp.abs(tokenization_error), axis=1), "tok_l1", action_dim
        ),
        **stats_for_metric(
            jnp.mean(jnp.square(tokenization_error), axis=1), "tok_l2", action_dim
        ),
        **stats_for_metric(
            pred_action_tokens == gt_action_tokens, "acc", tokenizer_config.num_action_tokens
        ),
    } | (
        {
            **stats_for_metric(
                optax.softmax_cross_entropy_with_integer_labels(
                    pred_action_logits, gt_action_tokens
                ),
                "action_loss",
                tokenizer_config.num_action_tokens
            ),
        }
        if pred_action_logits is not None
        else {}
    )

def get_value(pred_values, tokens, tokenizer_config: Tokenizer.TokenizerConfig):
    value_token_starts = jnp.argmax(tokens == tokenizer_config.end_of_action_token, axis=-1)
    _get_values = jax.vmap(
        lambda x, i: jax.lax.dynamic_slice(x, (i,), (1,))
    )
    qs = jax.vmap(_get_values, in_axes=(-1, None), out_axes=-1)(
        pred_values, value_token_starts
    ).squeeze()
    return qs

def compute_stats(
    *,
    detokenize_fn,
    pred_logits,
    tokens,
    actions,
    mask_loss,
    tokenizer_config: Tokenizer.TokenizerConfig,
    log_segment_prefix: str = "",
):
    output_pred_mask = mask_loss[..., 1:]
    labels = tokens[..., 1:]

    action_token_info = get_action_tokens(
        pred_logits, labels, tokenizer_config.num_action_tokens, tokenizer_config.begin_of_action_token
    )

    metrics = compute_action_metrics(
        detokenize_fn,
        **action_token_info,
        action_dim=actions.shape[-1],
        gt_actions=actions,
        tokenizer_config=tokenizer_config,
        log_segment_prefix=log_segment_prefix,
    )
    loss = jnp.mean(
        output_pred_mask
        * optax.softmax_cross_entropy_with_integer_labels(pred_logits, labels)
    ) / jnp.mean(output_pred_mask)
    metrics["loss"] = loss

    return loss, metrics


def step_fn(
    train_state: TrainState,
    batch: TrainingBatch,
    key: chex.PRNGKey,
    target_params,
    tokenizer_config: Tokenizer.TokenizerConfig,
    detokenize_fn,
    tokenize_fn,
    decode_fn,
    train: bool,
):
    def loss_fn(params, batch, key: chex.PRNGKey):
        all_inputs = batch.sensors | {"text": batch.tokens[..., :-1]}
        all_masks = batch.sensors_mask | {
            "text": jnp.ones_like(batch.tokens[..., :-1], dtype=jnp.bool_)
        }
        rng, key = jax.random.split(key)

        logits, info = train_state.apply_fn(
            {"params": params},
            all_inputs,
            data_masks=all_masks,
            text_ar_mask=batch.tokens_ar[..., :-1],
            train=train,
            rngs={"dropout": key},
        )

        actor_loss, actor_metrics = compute_stats(
            detokenize_fn=partial(detokenize_fn, obs=batch.sensors),
            pred_logits=logits,
            tokens=batch.tokens,
            actions=batch.actions,
            mask_loss=batch.tokens_loss,
            tokenizer_config=tokenizer_config,
            log_segment_prefix="train/tf_",
        )

        qs = get_value(info["values"], batch.tokens[..., 1:], tokenizer_config)
        value_logits = get_value(info["value_logits"], batch.tokens[..., 1:], tokenizer_config)
        del info

        scalar_target_to_dist_fn = hl_gauss_transform(
            min_value=-1.0 / (1 - 0.98),
            max_value=0.0 / (1 - 0.98),
            num_bins = 256,
        )[0]

        """
        CQL + HL Gauss
        """
        # TD target
        rollout_batch = RolloutBatch(
            sensor_data=batch.sensors_next,
            sensor_masks=batch.sensors_next_mask,
            prompt=batch.next_tokens,
            prompt_mask=batch.next_mask_input,
            prompt_ar=jnp.zeros_like(batch.next_mask_input),
        )
        
        _, q_pi_next = decode_fn(
            rollout_batch,
            target_key_order=None,
            params=target_params,
        )  # out_tokens: (batch_size, 7), value: (batch_size,)

        td_target = batch.rewards + batch.td_mask * q_pi_next * 0.98
        td_target = jax.lax.stop_gradient(td_target) # just in case
        td_loss = cross_entropy_loss_on_scalar(
            value_logits,
            td_target,
            scalar_target_to_dist_fn,
        ).mean()
        
        # # sample policy actions and compute  q_pi
        rollout_batch = RolloutBatch(
            sensor_data=batch.sensors,
            sensor_masks=batch.sensors_mask,
            prompt=batch.gen_tokens,
            prompt_mask=batch.gen_mask_input,
            prompt_ar=jnp.zeros_like(batch.tokens_ar),
        )
        policy_tokens, _ = decode_fn(
            rollout_batch,
            target_key_order=None,
            params=target_params, # need to use taret_params otherwie somehow there where be OOM error
        )
        policy_tokens = jax.lax.stop_gradient(policy_tokens) # just in case
        policy_tokens = replace_action_tokens(batch.tokens, policy_tokens, tokenizer_config.begin_of_action_token)
        rng, key = jax.random.split(rng)
        _, info = train_state.apply_fn(
            {"params": params},
            batch.sensors | {"text": policy_tokens[..., :-1]},
            data_masks=batch.sensors_mask | {
                "text": jnp.ones_like(policy_tokens[..., :-1], dtype=jnp.bool_)
            },
            text_ar_mask=batch.tokens_ar[..., :-1],
            train=True,
            rngs={"dropout": key},
        )
        q_pi = get_value(info["values"], policy_tokens[..., 1:], tokenizer_config)
        del info

        # # compute q_rand
        rng, key = jax.random.split(rng)
        random_actions = jax.random.uniform(
                key,
                shape=batch.actions.shape,
                minval=-1.0,
                maxval=1.0,
        )
        random_tokens = tokenize_fn(random_actions, obs=batch.sensors)
        random_tokens = replace_action_tokens(batch.tokens, random_tokens, tokenizer_config.begin_of_action_token)
        rng, key = jax.random.split(rng)
        _, info = train_state.apply_fn(
            {"params": params},
            batch.sensors | {"text": random_tokens[..., :-1]},
            data_masks=batch.sensors_next_mask | {
                "text": jnp.ones_like(random_tokens[..., :-1], dtype=jnp.bool_)
            },
            text_ar_mask=batch.tokens_ar[..., :-1],
            train=True,
            rngs={"dropout": key},
        )
        q_rand = get_value(info["values"], random_tokens[..., 1:], tokenizer_config)
        del info


        cql_cat_q = jnp.stack([q_rand, q_pi, q_pi_next, qs], axis=-1)

        # cql_cat_q = jnp.stack([q_pi, qs], axis=-1)
        lse_q = jax.scipy.special.logsumexp(cql_cat_q, axis=1)
        cql_loss = jnp.mean(lse_q - qs)

        cql_alpha = 5.0
        critic_loss = td_loss + cql_alpha * cql_loss

        critic_metrics = {
            "critic/critic_loss": critic_loss,
            "critic/td_loss": td_loss,
            "critic/cql_loss": cql_loss,
            "critic/q_gt": batch.mc_returns.mean(),
            "critic/q_pred": qs.mean(),
            "critic/td_target": td_target.mean(),
            "critic/q_pi_next": q_pi_next.mean(),
            "critic/q_pi": q_pi.mean(),
            "critic/q_rand": q_rand.mean(),
            "critic/lse_q": lse_q.mean(),
        }



        
        # HL Gauss + MC regression
        # target_q = batch.mc_returns
        # critic_loss = cross_entropy_loss_on_scalar(
        #     value_logits,
        #     target_q,
        #     scalar_target_to_dist_fn,
        # ).mean()
        # critic_metrics = {
        #     "critic/critic_loss": critic_loss,
        #     "critic/q_gt": batch.mc_returns.mean(),
        #     "critic/q_pred": qs.mean(),
        # }

        # HL Gauss + SARSA
        # _, next_value_info = train_state.apply_fn(
        #     {"params": target_params},
        #     batch.sensors_next | {"text": batch.next_tokens[..., :-1]},
        #     data_masks=batch.sensors_next_mask | {
        #         "text": jnp.ones_like(batch.next_tokens[..., :-1], dtype=jnp.bool_)
        #     },
        #     text_ar_mask=batch.tokens_ar[..., :-1],
        #     train=False,
        #     rngs={"dropout": key_value},
        # )

        # next_qs = next_value_info["values"]
        # next_qs = get_value(next_qs, batch.next_tokens[..., 1:], tokenizer_config)

        # td_target = batch.rewards + batch.td_mask * next_qs * 0.98
        # td_target = jax.lax.stop_gradient(td_target) # just in case

        # critic_loss = cross_entropy_loss_on_scalar(
        #     value_logits,
        #     td_target,
        #     scalar_target_to_dist_fn,
        # ).mean()
        # critic_metrics = {
        #     "critic/critic_loss": critic_loss,
        #     "critic/q_gt": batch.mc_returns.mean(),
        #     "critic/q_pred": qs.mean(),
        #     "critic/td_target": td_target.mean(),
        #     "critic/next_q_pred": next_qs.mean(),
        # }


        actor_metrics.update(critic_metrics)

        return actor_loss + critic_loss, actor_metrics

    grad_fn = jax.grad(loss_fn, has_aux=True)

    key, dropout_key = jax.random.split(key)
    grads, info = grad_fn(train_state.params, batch, dropout_key)
    updates, opt_state = train_state.tx.update(
        grads, train_state.opt_state, params=train_state.params
    )
    params = optax.apply_updates(train_state.params, updates)

    new_target_params = optax.incremental_update(
           params,
           target_params,
           step_size=0.005,
        )

    # new_target_params = train_state.soft_update_target(params, params, 0.005)

    train_state = train_state.replace(
        params=params, opt_state=opt_state, step=train_state.step + 1, 
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
        | _norm_info(new_target_params, "norm/target_param")
    )

    return train_state, info, key, new_target_params
