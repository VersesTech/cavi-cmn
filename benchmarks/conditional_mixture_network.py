# This code is part of the VersesTech Repository `cavi-cmn` (https://github.com/VersesTech/cavi-cmn).
# It is licensed under the VERSES Academic Research License.
#
# For more information, please refer to the license file:
# https://github.com/VersesTech/cavi-cmn/blob/main/license.txt

import jax
from jax import jit
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax import random as jr, nn, vmap
from jax.scipy.linalg import cho_solve

from functools import partial

import numpyro.distributions as dist
from numpyro.ops.indexing import Vindex
from numpyro.infer import MCMC, NUTS, Predictive, SVI, TraceEnum_ELBO, log_likelihood
from numpyro import subsample, sample, plate, deterministic, handlers, param

from .stickbreaking_util import log_stb, betas_init

import optax

from tensorflow_probability.substrates import jax as tfp

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


def linear_wishart_mnlr(
    layer,
    x_dim,
    y_dim,
    K,
    dof_offset,
    inv_V_scale,
    fixed_precision,
    prob_type,
    beta_scale,
):
    """
    Sample the parameters of a single Directed Mixture Layer.
    """
    if prob_type == "softmax":
        with plate(f"betas{layer}.classes", K):
            B = sample(
                f"layer{layer}.B",
                dist.Normal(0.0, beta_scale).expand([K, x_dim + 1]).to_event(1),
            )
    else:
        with plate(f"betas{layer}.classes", K - 1):
            B = sample(
                f"layer{layer}.B",
                dist.Normal(0.0, beta_scale).expand([K - 1, x_dim + 1]).to_event(1),
            )

    prior_n = (
        y_dim + 1 + dof_offset
    )  # the prior degrees of freedom parameter for the Wishart distribution
    scale_tril_U = 1.0  # this is the inverse of the scale we use to scale prior over the inverse of U via inv_U = (scale**2) * jnp.eye(y_dim)
    # scale_tril_U = (K**(0.5 /y_dim)) / jnp.sqrt(0.05) # this is the inverse of the scale we use to scale prior over the inverse of U via inv_U = (scale**2) * jnp.eye(y_dim)
    scale_tril_V = jnp.sqrt(1.0 / inv_V_scale)
    with plate(f"params{layer}", K):
        if fixed_precision:
            L = jnp.broadcast_to(
                scale_tril_U * jnp.eye(y_dim) * jnp.sqrt(prior_n), (K, y_dim, y_dim)
            )
        else:
            L = sample(
                f"layer{layer}.L",
                dist.WishartCholesky(prior_n, scale_tril=scale_tril_U * jnp.eye(y_dim)),
            )
        z = sample(
            f"layer{layer}.Z", dist.Normal(0, 1).expand([y_dim, x_dim + 1]).to_event(2)
        )

        L, I = jnp.broadcast_arrays(L, jnp.eye(y_dim))
        Sigma = cho_solve((L, True), I)
        L = jnp.linalg.cholesky(Sigma)
        A = deterministic(f"layer{layer}.A", (L @ (z * scale_tril_V)).mT)

    return (B.mT, A, L)


def linear_gamma_mnlr(
    layer,
    x_dim,
    y_dim,
    K,
    dof_offset,
    inv_V_scale,
    fixed_precision,
    prob_type,
    beta_scale,
):
    """
    Sample the parameters of a single Directed Mixture Layer.
    """
    if prob_type == "softmax":
        with plate(f"betas{layer}.classes", K):
            B = sample(
                f"layer{layer}.B",
                dist.Normal(0.0, beta_scale).expand([K, x_dim + 1]).to_event(1),
            )
    else:
        with plate(f"betas{layer}.classes", K - 1):
            B = sample(
                f"layer{layer}.B",
                dist.Normal(0.0, beta_scale).expand([K - 1, x_dim + 1]).to_event(1),
            )

    prior_a = (
        1 + dof_offset
    )  # the prior degrees of freedom parameter for the Wishart distribution
    scale_tril_U = 1.0  # this is the inverse of the scale we use to scale prior over the inverse of U via inv_U = (scale**2) * jnp.eye(y_dim)
    # scale_tril_U = (K**(0.5 /y_dim)) / jnp.sqrt(0.05) # this is the inverse of the scale we use to scale prior over the inverse of U via inv_U = (scale**2) * jnp.eye(y_dim)
    prior_b = scale_tril_U**2
    scale_tril_V = jnp.sqrt(1.0 / inv_V_scale)
    with plate(f"params{layer}", K):
        if fixed_precision:
            variance = jnp.broadcast_to(
                prior_b / (prior_a - 1),
                (
                    K,
                    y_dim,
                ),
            )
        else:
            variance = sample(
                f"layer{layer}.variance",
                dist.InverseGamma(prior_a, prior_b).expand([y_dim]).to_event(1),
            )

        z = sample(
            f"layer{layer}.Z", dist.Normal(0, 1).expand([y_dim, x_dim + 1]).to_event(2)
        )
        std = jnp.sqrt(variance)
        L = deterministic(f"layer{layer}.inv_sigma", vmap(jnp.diag)(std))
        A = deterministic(
            f"layer{layer}.A", (jnp.expand_dims(std, -1) * (z * scale_tril_V)).mT
        )
        # A = deterministic(f'layer{layer}.A', (z * scale_tril_V).mT )

    return (B.mT, A, L)


def cmix_layer_latents(data_plate, layer, X, B, A, L, prob_type, y=None):
    """
    Sample the latents of a single Conditional Mixture Layer.
    """
    with data_plate:
        x_batch = subsample(X, event_dim=1)
        y_batch = None if y is None else subsample(y, event_dim=1)

        logits = x_batch @ B[:-1] + B[-1]
        if prob_type == "stick-breaking":
            logits = log_stb(logits)

        z = sample(
            "z", dist.Categorical(logits=logits), infer={"enumerate": "parallel"}
        )
        A = Vindex(A)
        loc = (jnp.expand_dims(x_batch, -2) @ A[z, :-1]).squeeze(-2) + A[z, -1]
        L = Vindex(L)[z]
        return sample(
            f"layer{layer}.y",
            dist.MultivariateNormal(loc=loc, scale_tril=L),
            obs=y_batch,
        )


def cmix_layer(
    data_plate,
    layer,
    x,
    y_dim,
    K,
    dof_offset,
    inv_V_scale,
    scale_beta,
    y=None,
    fixed_precision=False,
    prob_type="stick-breaking",
    prior_type="wishart",
):
    """
    Single Conditional Mixture Layer.
    """
    _, x_dim = x.shape

    if prior_type == "wishart":
        params = linear_wishart_mnlr(
            layer,
            x_dim,
            y_dim,
            K,
            dof_offset,
            inv_V_scale,
            fixed_precision,
            prob_type,
            scale_beta,
        )
    else:
        params = linear_gamma_mnlr(
            layer,
            x_dim,
            y_dim,
            K,
            dof_offset,
            inv_V_scale,
            fixed_precision,
            prob_type,
            scale_beta,
        )

    return cmix_layer_latents(data_plate, layer, x, *params, prob_type, y=y)


def mnlr_output(
    data_plate, X, num_classes, scale_beta, y=None, prob_type="stick-breaking"
):
    """
    Multinomial Logistic Regression output layer.
    """
    _, x_dim = X.shape

    if prob_type == "softmax":
        with plate(f"betas.output.labels", num_classes):
            B = sample(
                "output.B",
                dist.Normal(0.0, scale_beta)
                .expand([num_classes, x_dim + 1])
                .to_event(1),
            )
    else:
        if num_classes > 2:
            with plate(f"betas.output.labels", num_classes - 1):
                B = sample(
                    "output.B",
                    dist.Normal(0.0, scale_beta)
                    .expand([num_classes - 1, x_dim + 1])
                    .to_event(1),
                )
        else:
            B = sample(
                "output.B", dist.Normal(0.0, scale_beta).expand([x_dim + 1]).to_event(1)
            )
            B = jnp.expand_dims(B, -2)

    with data_plate:
        x_batch = subsample(X, event_dim=1)
        y_batch = None if y is None else subsample(y, event_dim=0)

        logits = x_batch @ B[:, :-1].mT + B[:, -1]

        if prob_type == "stick-breaking":
            logits = log_stb(logits)

        deterministic("output.logits", logits)
        sample("obs", dist.Categorical(logits=logits), obs=y_batch)


def multilayer_cmn(
    x,
    layer_dims,
    num_components,
    y=None,
    num_classes=2,
    dof_offset=1.0,
    inv_V_scale=1e-1,
    scale_beta=1.0,
    batch_size=None,
    fixed_precision=False,
    prob_type="stick-breaking",
    prior_type="wishart",
):
    """
    Conditional Mixture Network with a Multinomial Logistic Regression output layer.
    """
    N, _ = x.shape
    data_plate = plate("data", N, subsample_size=batch_size)
    for l, (y_dim, k) in enumerate(zip(layer_dims, num_components)):
        x = cmix_layer(
            data_plate,
            l,
            x,
            y_dim=y_dim,
            K=k,
            dof_offset=dof_offset,
            inv_V_scale=inv_V_scale,
            scale_beta=scale_beta,
            fixed_precision=fixed_precision,
            prob_type=prob_type,
            prior_type=prior_type,
        )

    mnlr_output(data_plate, x, num_classes, scale_beta, y=y, prob_type=prob_type)


def mnlr_output_guide(x_dim, num_classes, prob_type):
    if prob_type == "softmax":
        loc = param(
            "beta.output.loc", lambda key: betas_init(key, (x_dim, num_classes))
        ).mT
        scale_tril = param(
            "beta.output.st",
            jnp.broadcast_to(
                jnp.eye(x_dim + 1) / 10, (num_classes, x_dim + 1, x_dim + 1)
            ),
            constraint=dist.constraints.softplus_lower_cholesky,
        )
        with plate(f"betas.output.labels", num_classes):
            B = sample("output.B", dist.MultivariateNormal(loc, scale_tril=scale_tril))
    else:
        loc = param(
            "beta.output.loc", lambda key: betas_init(key, (x_dim, num_classes - 1))
        ).mT
        scale_tril = param(
            "beta.output.st",
            jnp.broadcast_to(
                jnp.eye(x_dim + 1) / 10, (num_classes - 1, x_dim + 1, x_dim + 1)
            ),
            constraint=dist.constraints.softplus_lower_cholesky,
        )

        if num_classes > 2:
            with plate(f"betas.output.labels", num_classes - 1):
                sample("output.B", dist.MultivariateNormal(loc, scale_tril=scale_tril))

        else:
            sample(
                "output.B",
                dist.MultivariateNormal(
                    loc.squeeze(0), scale_tril=scale_tril.squeeze(0)
                ),
            )


def guide_wishart(layer, x_dim, y_dim, K, N, prob_type):
    """
    Approximate posterior of a single Directed Mixture Layer.
    """
    i_x = jnp.eye(x_dim + 1)
    i_y = jnp.eye(y_dim)
    if prob_type == "softmax":
        loc = param(f"betas{layer}.loc", lambda key: betas_init(key, (x_dim, K))).mT
        scale_tril = param(
            f"betas{layer}.st",
            jnp.broadcast_to(i_x / 10, (K, x_dim + 1, x_dim + 1)),
            constraint=dist.constraints.softplus_lower_cholesky,
        )
        with plate(f"betas{layer}.classes", K):
            sample(
                f"layer{layer}.B", dist.MultivariateNormal(loc, scale_tril=scale_tril)
            )
    else:
        loc = param(f"betas{layer}.loc", lambda key: betas_init(key, (x_dim, K - 1))).mT
        scale_tril = param(
            f"betas{layer}.st",
            jnp.broadcast_to(i_x / 10, (K - 1, x_dim + 1, x_dim + 1)),
            constraint=dist.constraints.softplus_lower_cholesky,
        )
        with plate(f"betas{layer}.classes", K - 1):
            sample(
                f"layer{layer}.B", dist.MultivariateNormal(loc, scale_tril=scale_tril)
            )

    df = param(
        f"layer{layer}.df",
        jnp.full((K,), y_dim + 1 + N / K),
        constraint=dist.constraints.softplus_positive,
    )
    st_pi = param(
        f"layer{layer}.st_pi",
        jnp.broadcast_to(i_y * 10, (K, y_dim, y_dim)),
        constraint=dist.constraints.softplus_lower_cholesky,
    )

    loc = param(
        f"Z{layer}.loc",
        lambda key: jr.uniform(
            key, shape=(K, y_dim, x_dim + 1), minval=-1.0, maxval=1.0
        ),
    )
    A = param(
        f"Z{layer}.A",
        jnp.broadcast_to(i_y / jnp.sqrt(10), (K, y_dim, y_dim)),
        constraint=dist.constraints.softplus_lower_cholesky,
    )
    B = param(
        f"Z{layer}.B",
        jnp.broadcast_to(i_x / jnp.sqrt(10), (K, x_dim + 1, x_dim + 1)),
        constraint=dist.constraints.softplus_lower_cholesky,
    )
    with plate(f"params{layer}", K):
        sample(f"layer{layer}.L", dist.WishartCholesky(df, scale_tril=st_pi))
        sample(f"layer{layer}.Z", dist.MatrixNormal(loc, A, B))


def guide_gamma(layer, x_dim, y_dim, K, N, prob_type):
    """
    Approximate posterior of a single Directed Mixture Layer.
    """
    i_x = jnp.eye(x_dim + 1)
    i_y = jnp.eye(y_dim)
    if prob_type == "softmax":
        loc = param(f"betas{layer}.loc", lambda key: betas_init(key, (x_dim, K))).mT
        scale_tril = param(
            f"betas{layer}.st",
            jnp.broadcast_to(i_x / 10, (K, x_dim + 1, x_dim + 1)),
            constraint=dist.constraints.softplus_lower_cholesky,
        )
        with plate(f"betas{layer}.classes", K):
            sample(
                f"layer{layer}.B", dist.MultivariateNormal(loc, scale_tril=scale_tril)
            )
    else:
        loc = param(f"betas{layer}.loc", lambda key: betas_init(key, (x_dim, K - 1))).mT
        scale_tril = param(
            f"betas{layer}.st",
            jnp.broadcast_to(i_x / 10, (K - 1, x_dim + 1, x_dim + 1)),
            constraint=dist.constraints.softplus_lower_cholesky,
        )
        with plate(f"betas{layer}.classes", K - 1):
            sample(
                f"layer{layer}.B", dist.MultivariateNormal(loc, scale_tril=scale_tril)
            )

    a = param(
        f"var{layer}.a",
        jnp.ones((K, y_dim)) * N / K,
        constraint=dist.constraints.softplus_positive,
    )
    b = param(
        f"var{layer}.b",
        jnp.ones((K, y_dim)),
        constraint=dist.constraints.softplus_positive,
    )

    loc = param(
        f"Z{layer}.loc",
        lambda key: jr.uniform(
            key, shape=(K, y_dim, x_dim + 1), minval=-1.0, maxval=1.0
        ),
    )
    A = param(
        f"Z{layer}.A",
        jnp.broadcast_to(i_y / jnp.sqrt(10), (K, y_dim, y_dim)),
        constraint=dist.constraints.softplus_lower_cholesky,
    )
    B = param(
        f"Z{layer}.B",
        jnp.broadcast_to(i_x / jnp.sqrt(10), (K, x_dim + 1, x_dim + 1)),
        constraint=dist.constraints.softplus_lower_cholesky,
    )

    with plate(f"params{layer}", K):
        sample(f"layer{layer}.variance", dist.InverseGamma(a, b).to_event(1))

        sample(f"layer{layer}.Z", dist.MatrixNormal(loc, A, B))


def guide_latents(data_plate, layer, x_dim, K, N, y=None):
    """
    Sample the latents of a single Directed Mixture Layer.
    """
    with data_plate:
        logits = param(f"layer{layer}.cats", jnp.zeros((N, K)))
        sample("z", dist.Categorical(logits=logits), infer={"enumerate": "parallel"})
        if y is None:
            loc = param(
                f"y{layer}.loc",
                lambda key: jr.uniform(key, shape=(N, x_dim), minval=-1.0, maxval=1.0),
            )
            # scale_tril = param(f'y{layer}.st', jnp.broadcast_to(jnp.eye(x_dim) / 10, (N, x_dim, x_dim)), constraint=dist.constraints.softplus_lower_cholesky)
            # sample(f"layer{layer}.y", dist.MultivariateNormal(loc=loc, scale_tril=scale_tril))
            scale = param(
                f"y{layer}.scale",
                jnp.ones((N, x_dim)) / 10,
                constraint=dist.constraints.softplus_positive,
            )
            sample(f"layer{layer}.y", dist.Normal(loc=loc, scale=scale).to_event(1))


def full_guide(
    x,
    layer_dims,
    num_components,
    y=None,
    num_classes=2,
    batch_size=None,
    prob_type="stick-breaking",
    prior_type="wishart",
    **kwargs,
):
    """
    Approximate posterior for the multilayer conditional mixture model with a Multinomial Logistic Regression output layer.
    """
    N, x_dim = x.shape
    data_plate = plate("data", N, subsample_size=batch_size)
    for l, (y_dim, k) in enumerate(zip(layer_dims, num_components)):
        if prior_type == "wishart":
            guide_wishart(l, x_dim, y_dim, k, N, prob_type)
        else:
            guide_gamma(l, x_dim, y_dim, k, N, prob_type)

        obs = None
        if y is not None:
            if isinstance(y, dict):
                if l in y:
                    # if latent is observed turn off posterior estimate
                    obs = True

        guide_latents(data_plate, l, y_dim, k, N, y=obs)
        x_dim = y_dim

    mnlr_output_guide(x_dim, num_classes, prob_type)


def marginalize_logits(
    x,
    prob_type,
    prior_type,
    layer_dims,
    num_components,
    num_classes,
    key,
    post_samples,
    ns=100,
    **kwargs,
):
    model = handlers.condition(multilayer_cmn, data=post_samples)
    pred = Predictive(model, num_samples=ns)

    # key, _key = jr.split(key)
    pred_samples = pred(
        key,
        x,
        layer_dims,
        num_components,
        num_classes=num_classes,
        prob_type=prob_type,
        prior_type=prior_type,
        **kwargs,
    )
    logits = pred_samples["output.logits"]
    logits = logsumexp(logits - jnp.log(ns), axis=0)

    return logits


def fit_cmn_bbvi(
    key,
    x_train,
    y_train,
    x_test,
    y_test,
    num_classes,
    layer_dims,
    num_components,
    prior_type,
    scale_beta=5.0,
    lr=5e-3,
    num_iters=20000,
    prob_type="stick-breaking",
    num_svi_samples=64,
    grid=None,
):

    loss = TraceEnum_ELBO(num_particles=8)
    optim = optax.adabelief(learning_rate=lr)

    svi = SVI(multilayer_cmn, full_guide, optim, loss)

    key, _key = jr.split(key)
    svi_res = svi.run(
        _key,
        num_iters,
        x_train,
        layer_dims,
        num_components,
        progress_bar=False,
        stable_update=True,
        scale_beta=scale_beta,
        y=y_train,
        num_classes=num_classes,
        prob_type=prob_type,
        prior_type=prior_type,
    )

    pred = Predictive(full_guide, params=svi_res.params, num_samples=num_svi_samples)

    key, _key = jr.split(key)
    post_svi_samples = pred(
        _key,
        x_train,
        layer_dims,
        num_components,
        scale_beta=scale_beta,
        num_classes=num_classes,
        prob_type=prob_type,
        prior_type=prior_type,
    )
    post_svi_samples.pop("layer0.y")
    post_svi_samples.pop("z")

    # train accuracy
    pred = Predictive(
        multilayer_cmn,
        posterior_samples=post_svi_samples,
        return_sites=["output.logits"],
    )
    key, _key = jr.split(key)
    smpls = pred(
        _key,
        x_train,
        layer_dims,
        num_components,
        scale_beta=scale_beta,
        num_classes=num_classes,
        prob_type=prob_type,
        prior_type=prior_type,
    )

    train_logits = smpls.pop("output.logits")

    train_logits = logsumexp(train_logits - jnp.log(num_svi_samples), axis=0)
    train_logits = train_logits - logsumexp(train_logits, -1, keepdims=True)
    y_pred = train_logits.argmax(-1)
    train_acc = jnp.mean(y_pred == y_train)

    # test accuracy
    logits_fn = partial(
        marginalize_logits,
        x_test,
        prob_type,
        prior_type,
        layer_dims,
        num_components,
        num_classes,
        ns=32,
        scale_beta=scale_beta,
    )

    keys = jr.split(key, num_svi_samples + 1)
    key = keys[-1]
    logits = vmap(logits_fn)(keys[:-1], post_svi_samples)

    logits = logsumexp(logits - jnp.log(num_svi_samples), axis=0)
    logits = logits - logsumexp(logits, -1, keepdims=True)
    y_pred = logits.argmax(-1)

    test_acc = jnp.mean(y_pred == y_test)
    lpd = jnp.sum(logits * nn.one_hot(y_test, num_classes), -1).mean()
    ece = tfp.stats.expected_calibration_error(
        20, logits=logits, labels_true=y_test, labels_predicted=y_pred
    )

    if grid is not None:
        logits_fn = partial(
            marginalize_logits,
            grid,
            prob_type,
            prior_type,
            layer_dims,
            num_components,
            num_classes,
            ns=32,
            scale_beta=scale_beta,
        )

        keys = jr.split(key, num_svi_samples + 1)
        key = keys[-1]
        logits = vmap(logits_fn)(keys[:-1], post_svi_samples)

        logits = logsumexp(logits - jnp.log(num_svi_samples), axis=0)
        logits = logits - logsumexp(logits, -1, keepdims=True)
        grid_predicted_class = logits.argmax(-1)
    else:
        grid_predicted_class = jnp.nan

    return train_acc, test_acc, lpd, ece, svi_res, grid_predicted_class


def compute_log_likelihood_bbvi(
    key,
    x,
    y,
    num_classes,
    layer_dims,
    num_components,
    prior_type,
    scale_beta=5.0,
    lr=5e-3,
    num_iters=20000,
    prob_type="stick-breaking",
    num_svi_samples=64,
):

    loss = TraceEnum_ELBO(num_particles=8)
    optim = optax.adabelief(learning_rate=lr)

    svi = SVI(multilayer_cmn, full_guide, optim, loss)

    key, _key = jr.split(key)
    svi_res = svi.run(
        _key,
        num_iters,
        x,
        layer_dims,
        num_components,
        progress_bar=False,
        stable_update=True,
        scale_beta=scale_beta,
        y=y,
        num_classes=num_classes,
        prob_type=prob_type,
        prior_type=prior_type,
    )

    pred = Predictive(full_guide, params=svi_res.params, num_samples=num_svi_samples)

    key, _key = jr.split(key)
    post_svi_samples = pred(
        _key,
        x,
        layer_dims,
        num_components,
        scale_beta=scale_beta,
        num_classes=num_classes,
        prob_type=prob_type,
        prior_type=prior_type,
    )

    return log_likelihood(
        multilayer_cmn,
        post_svi_samples,
        x,
        layer_dims,
        num_components,
        scale_beta=scale_beta,
        y=y,
        num_classes=num_classes,
        prob_type=prob_type,
        prior_type=prior_type,
    )


def fit_cmn_nuts(
    key,
    x_train,
    y_train,
    x_test,
    y_test,
    num_classes,
    layer_dims,
    num_components,
    prior_type,
    scale_beta=5.0,
    num_warmup=800,
    num_nuts_samples=64,
    num_chains=16,
    prob_type="stick-breaking",
    grid=None,
):

    kernel = NUTS(multilayer_cmn)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_nuts_samples,
        num_chains=num_chains,
        chain_method="vectorized",
        progress_bar=False,
    )

    key, _key = jr.split(key)
    mcmc.run(
        _key,
        x_train,
        layer_dims,
        num_components,
        y=y_train,
        scale_beta=scale_beta,
        num_classes=num_classes,
        prob_type=prob_type,
        prior_type=prior_type,
    )

    post_samples = mcmc.get_samples()
    post_samples.pop("layer0.y")
    train_logits = post_samples.pop("output.logits").reshape(
        num_chains, num_nuts_samples, len(x_train), num_classes
    )

    train_logits = logsumexp(train_logits - jnp.log(num_nuts_samples), axis=1)
    train_logits = train_logits - logsumexp(train_logits, -1, keepdims=True)
    y_pred = train_logits.argmax(-1)

    train_acc = jnp.mean(y_pred == y_train, axis=-1)

    # test accuracy
    logits_fn = partial(
        marginalize_logits,
        x_test,
        prob_type,
        prior_type,
        layer_dims,
        num_components,
        num_classes,
        ns=32,
        scale_beta=scale_beta,
    )

    keys = jr.split(key, num_nuts_samples * num_chains + 1)
    key = keys[-1]
    logits = vmap(logits_fn)(keys[:-1], post_samples).reshape(
        num_chains, num_nuts_samples, len(x_test), num_classes
    )

    logits = logsumexp(logits - jnp.log(num_nuts_samples), axis=1)
    logits = logits - logsumexp(logits, -1, keepdims=True)
    y_pred = logits.argmax(-1)

    test_acc = jnp.mean(y_pred == y_test, axis=-1)
    lpd = jnp.sum(logits * nn.one_hot(y_test, num_classes), -1).mean(-1)

    ece_fn = lambda x, y: tfp.stats.expected_calibration_error(
        20, logits=x, labels_true=y_test, labels_predicted=y
    )
    ece = vmap(ece_fn)(logits, y_pred)

    if grid is not None:
        logits_fn = partial(
            marginalize_logits,
            grid,
            prob_type,
            prior_type,
            layer_dims,
            num_components,
            num_classes,
            ns=32,
            scale_beta=scale_beta,
        )

        keys = jr.split(key, num_nuts_samples * num_chains + 1)
        key = keys[-1]
        logits = vmap(logits_fn)(keys[:-1], post_samples)

        logits = logsumexp(logits - jnp.log(num_nuts_samples * num_chains), axis=0)
        logits = logits - logsumexp(logits, -1, keepdims=True)
        grid_predicted_class = logits.argmax(-1)
    else:
        grid_predicted_class = jnp.nan

    return train_acc, test_acc, lpd, ece, grid_predicted_class


def compute_log_likelihood_nuts(
    key,
    x,
    y,
    num_classes,
    layer_dims,
    num_components,
    prior_type,
    scale_beta=5.0,
    num_warmup=800,
    num_nuts_samples=64,
    num_chains=16,
    prob_type="stick-breaking",
):

    kernel = NUTS(multilayer_cmn)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_nuts_samples,
        num_chains=num_chains,
        chain_method="vectorized",
        progress_bar=False,
    )

    key, _key = jr.split(key)
    mcmc.run(
        _key,
        x,
        layer_dims,
        num_components,
        y=y,
        scale_beta=scale_beta,
        num_classes=num_classes,
        prob_type=prob_type,
        prior_type=prior_type,
    )

    handler_seed = jr.randint(
        _key, shape=(), minval=0, maxval=2**15, dtype=jnp.int32
    ).item()
    post_samples = mcmc.get_samples(group_by_chain=True)
    return log_likelihood(
        handlers.seed(multilayer_cmn, rng_seed=handler_seed),
        post_samples,
        x,
        layer_dims,
        num_components,
        scale_beta=scale_beta,
        batch_ndims=2,
        y=y,
        num_classes=num_classes,
        prob_type=prob_type,
        prior_type=prior_type,
    )
