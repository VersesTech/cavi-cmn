from jax import numpy as jnp
from jax import random as jr
from jax import nn
from jax.scipy.special import logsumexp
from .stickbreaking_util import log_stb, betas_init

from numpyro import sample, subsample, plate, deterministic, param
import numpyro.distributions as dist
from numpyro.infer import Predictive, SVI, TraceEnum_ELBO, Trace_ELBO
from tensorflow_probability.substrates import jax as tfp

import optax


def network_params(layer, x_dim, y_dim, K, prob_type="softmax"):
    """
    Sample the initial parameters of the Mixture-of-Experts network
    """
    B_fn = lambda d: param(f"layer{layer}.B", lambda key: betas_init(key, (x_dim, d)))
    B = B_fn(K) if prob_type == "softmax" else B_fn(K - 1)
    A = param(
        f"layer{layer}.A",
        lambda key: jr.uniform(
            key,
            shape=(x_dim + 1, y_dim, K),
            minval=-1 / jnp.sqrt(x_dim + 1),
            maxval=1 / jnp.sqrt(x_dim + 1),
        ),
    )

    return (B, A)


def cmix_layer(x_batch, B, A, prob_type="softmax"):
    """
    Compute the forward pass of a conditional mixture layer, framed as a deterministic MoE forward pass.
    1. Use the input regressors `x_batch` and the regression coefficients `B` to compute the logits of the gating network.
    2. Use the input regressors `x_batch` and the regression coefficients `A` to compute the linear predictions of each expert
    3. Combine the predictions of each expert using the gating network's output probabilities
    """
    logits = x_batch @ B[:-1] + B[-1]
    if prob_type == "stick-breaking":
        logits = log_stb(logits)

    probs = nn.softmax(logits, -1)
    loc = jnp.moveaxis(x_batch @ jnp.moveaxis(A[:-1], -1, 0), 0, -1) + A[-1]

    return jnp.sum(loc * jnp.expand_dims(probs, -2), axis=-1)


def multilayer_cmn_mle(
    x,
    layer_dims,
    num_components,
    y=None,
    num_classes=2,
    batch_size=None,
    prob_type="stick-breaking",
):
    """
    Multi-layer Conditional Mixture Model with a Multinomial Logistic Regression output layer.
    """

    N, _ = x.shape
    with plate("data", N, subsample_size=batch_size):
        x_batch = subsample(x, event_dim=1)
        y_batch = None if y is None else subsample(y, event_dim=1)

        for l, (y_dim, k) in enumerate(zip(layer_dims, num_components)):
            params = network_params(l, x_batch.shape[-1], y_dim, k, prob_type=prob_type)
            x_batch = cmix_layer(x_batch, *params, prob_type=prob_type)

        # Multinomial Logistic Regression output.
        B_fn = lambda d: param(
            f"ouput.B", lambda key: betas_init(key, (x_batch.shape[-1], d))
        )
        B = B_fn(num_classes) if prob_type == "softmax" else B_fn(num_classes - 1)
        logits = x_batch @ B[:-1] + B[-1]
        if prob_type == "stick-breaking":
            logits = log_stb(logits)

        deterministic("output.logits", nn.softmax(logits))
        sample("obs", dist.Categorical(logits=logits), obs=y_batch)


def mle_guide(*args, **kwargs):
    pass


def fit_cmn_maximum_likelihood(
    key,
    x_train,
    y_train,
    x_test,
    y_test,
    num_classes,
    layer_dims,
    num_components,
    lr=1e-3,
    num_iters=5000,
    prob_type="stick-breaking",
    grid=None,
):

    loss = Trace_ELBO()
    optim = optax.adabelief(learning_rate=lr)

    svi = SVI(multilayer_cmn_mle, mle_guide, optim, loss)

    key, _key = jr.split(key)
    res = svi.run(
        _key,
        num_iters,
        x_train,
        layer_dims,
        num_components,
        progress_bar=False,
        y=y_train,
        num_classes=num_classes,
        prob_type=prob_type,
    )

    key, _key = jr.split(key)
    pred = Predictive(multilayer_cmn_mle, params=res.params, num_samples=1)
    svi_train = pred(
        _key,
        x_train,
        layer_dims,
        num_components,
        num_classes=num_classes,
        prob_type=prob_type,
    )
    train_logits = svi_train["output.logits"][0]
    train_logits = train_logits - logsumexp(train_logits, -1, keepdims=True)
    y_train_pred = train_logits.argmax(-1)
    train_acc = jnp.mean(y_train_pred == y_train)

    key, _key = jr.split(key)
    svi_test = pred(
        _key,
        x_test,
        layer_dims,
        num_components,
        num_classes=num_classes,
        prob_type=prob_type,
    )
    test_logits = svi_test["output.logits"][0]
    test_logits = test_logits - logsumexp(test_logits, -1, keepdims=True)
    y_test_pred = test_logits.argmax(-1)
    test_acc = jnp.mean(y_test_pred == y_test)

    lpd = jnp.sum(test_logits * nn.one_hot(y_test, num_classes), -1).mean()
    ece = tfp.stats.expected_calibration_error(
        20, logits=test_logits, labels_true=y_test, labels_predicted=y_test_pred
    )

    if grid is not None:
        key, _key = jr.split(key)
        svi_grid = pred(
            _key,
            grid,
            layer_dims,
            num_components,
            num_classes=num_classes,
            prob_type=prob_type,
        )
        grid_predicted_class = svi_grid["output.logits"][0].argmax(-1)
    else:
        grid_predicted_class = jnp.nan
    return train_acc, test_acc, lpd, ece, res, grid_predicted_class


def compute_log_likelihood_cmn(
    key,
    x,
    y,
    num_classes,
    layer_dims,
    num_components,
    lr=1e-3,
    num_iters=5000,
    prob_type="stick-breaking",
):

    loss = Trace_ELBO()
    optim = optax.adabelief(learning_rate=lr)

    svi = SVI(multilayer_cmn_mle, mle_guide, optim, loss)

    key, _key = jr.split(key)
    res = svi.run(
        _key,
        num_iters,
        x,
        layer_dims,
        num_components,
        progress_bar=False,
        y=y,
        num_classes=num_classes,
        prob_type=prob_type,
    )

    key, _key = jr.split(key)
    pred = Predictive(multilayer_cmn_mle, params=res.params, num_samples=1)
    svi_all_data = pred(
        _key,
        x,
        layer_dims,
        num_components,
        num_classes=num_classes,
        prob_type=prob_type,
    )
    logits = svi_all_data["output.logits"][0]
    logits = logits - logsumexp(logits, -1, keepdims=True)

    return jnp.sum(logits * nn.one_hot(y, num_classes), -1).mean()
