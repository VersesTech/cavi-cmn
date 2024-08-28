from jax import numpy as jnp
from jax import random as jr
from jax import nn


def log_stb(x):
    """Compute the log of the stick-breaking probabilities."""

    x = x - jnp.log(x.shape[-1] - jnp.arange(x.shape[-1]))
    # convert to probabilities (relative to the remaining) of each fraction of the stick
    sp = nn.softplus(x)
    log_z = x - sp
    log_z1m_cumprod = jnp.cumsum(-sp, axis=-1)
    pad_width = [(0, 0)] * x.ndim
    pad_width[-1] = (0, 1)
    log_z_padded = jnp.pad(log_z, pad_width)
    pad_width = [(0, 0)] * x.ndim
    pad_width[-1] = (1, 0)
    log_z1m_cumprod_shifted = jnp.pad(log_z1m_cumprod, pad_width)
    return log_z_padded + log_z1m_cumprod_shifted


def betas_init(key, shape):
    """
    Initialize the betas (regression coefficients) for the stick-breaking transform that maps from continuous
    input regressors to output class probabilities.
    """
    return jnp.pad(
        jr.uniform(key, shape=shape, minval=-1 / jnp.sqrt(shape[0]), maxval=1 / jnp.sqrt(shape[0])), [(0, 1), (0, 0)]
    )
