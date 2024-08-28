# This code is part of the VersesTech Repository `cavi-cmn` (https://github.com/VersesTech/cavi-cmn).
# It is licensed under the VERSES Academic Research License.
#
# For more information, please refer to the license file:
# https://github.com/VersesTech/cavi-cmn/blob/main/license.txt

import jax.numpy as jnp
from jax import random as jr, lax
from multimethod import multimethod
from jaxtyping import Array
from typing import Union

from .utils import ArrayDict, tree_at
from .distribution import Distribution, Delta
from .exponential import Multinomial, MixtureMessage, MultivariateNormal
from .models import Sequential, ConditionalMixture, ConditionalMixtureNetwork
from .transforms import MultinomialRegression


def cmix_layer(
    linear_key,
    mnlr_key,
    batch_shape,
    n_components,
    x_dim,
    y_dim,
    scale=1.0,
    dof_offset=1.0,
    inv_v_scale=1e-1,
    learning_rate_linear=1.0,
    batch_decay_linear=0.0,
    n_iters_mnlr=1,
    learning_rate_mnlr=1.0,
    batch_decay_mnlr=0.0,
    use_bias=True,
    fixed_precision=False,
    likelihood_prior_type="mnw",
    scale_mnlr_betas=1.0,
    **other_pi_kwargs
):
    """Initialize a directed mixture of linear transforms layer"""
    likelihood_params = {
        "args": (),
        "kwargs": {
            "batch_shape": batch_shape + (n_components,),
            "event_shape": (y_dim, x_dim + 1),
            "use_bias": use_bias,
            "init_key": linear_key,
            "scale": scale,
            "dof_offset": dof_offset,
            "inv_v_scale": inv_v_scale,
            "fixed_precision": fixed_precision,
        },
    }

    pi_kwargs = {
        "batch_shape": batch_shape,
        "rng_key": mnlr_key,
        "scale": scale_mnlr_betas,
    }
    # add other_pi_kwargs to pi_kwargs
    pi_kwargs.update(other_pi_kwargs)

    pi_params = {
        "args": [x_dim, n_components],
        "kwargs": pi_kwargs,
    }  # x_dim, y_dim arguments for MNLR
    return ConditionalMixture(
        likelihood_params=likelihood_params,
        pi_params=pi_params,
        likelihood_opts={"lr": learning_rate_linear, "beta": batch_decay_linear},
        pi_opts={
            "iters": n_iters_mnlr,
            "lr": learning_rate_mnlr,
            "beta": batch_decay_mnlr,
        },
        average_type="nat_params",
        likelihood_prior_type=likelihood_prior_type,
    )


def mnlr_layer(mnlr_key, batch_shape, x_dim, y_dim, **kwargs):
    """
    Initialize an MNLR layer
    """

    mnlr = MultinomialRegression(
        x_dim, y_dim, batch_shape=batch_shape, rng_key=mnlr_key, **kwargs
    )

    return mnlr


def initialize_network(
    mnlr_keys,
    linear_keys,
    n_components,
    hidden_dims,
    batch_shape,
    dof_offset,
    inv_v_scale,
    x_dim,
    y_dim,
    add_mnlr_output,
    n_vb_iters_mnlr,
    scale_likelihood,
    scale_mnlr_betas,
    init_posterior_scale,
    sample_initial_betas,
    fixed_precision,
    likelihood_type,
    n_backwards_iters,
    cmix_optim_args,
    mnlr_optim_args,
    compute_elbo=True,
):
    layers = []
    for layer_i, (nc, hd) in enumerate(zip(n_components, hidden_dims[:-1])):
        if layer_i == 0:
            in_dim, out_dim = x_dim, hd
        elif layer_i >= 1:
            in_dim, out_dim = hidden_dims[layer_i - 1], hd

        layer = cmix_layer(
            linear_keys[layer_i],
            mnlr_keys[layer_i],
            batch_shape,
            nc,
            in_dim,
            out_dim,
            scale=scale_likelihood,
            dof_offset=dof_offset,
            inv_v_scale=inv_v_scale,
            n_iters_mnlr=n_vb_iters_mnlr[layer_i],
            fixed_precision=fixed_precision,
            likelihood_prior_type=likelihood_type,
            scale_mnlr_betas=scale_mnlr_betas[layer_i],
            init_posterior_scale=init_posterior_scale[layer_i],
            sample_initial_betas=sample_initial_betas[layer_i],
            **cmix_optim_args,
        )
        layers.append(layer)

    if add_mnlr_output:
        layer = mnlr_layer(
            mnlr_keys[-1],
            batch_shape,
            hidden_dims[-1],
            y_dim,
            scale=scale_mnlr_betas[-1],
            init_posterior_scale=init_posterior_scale[-1],
            sample_initial_betas=sample_initial_betas[-1],
            optim_args=mnlr_optim_args,
        )
        layers.append(layer)

    return ConditionalMixtureNetwork(
        layers=layers,
        n_backwards_iters=n_backwards_iters,
        backwards_type="smooth",
        compute_elbo=compute_elbo,
    )
