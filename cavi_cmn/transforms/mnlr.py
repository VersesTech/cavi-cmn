# Copyright 2024 VERSES AI, Inc.
#
# Licensed under the VERSES Academic Research License (the “License”);
# you may not use this file except in compliance with the license.
#
# You may obtain a copy of the License at
#
#     https://github.com/VersesTech/cavi-cmn/blob/main/license.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Tuple, Union, List
from functools import partial
from jaxtyping import Array, PRNGKeyArray

from multimethod import multimethod

import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import logsumexp

from jax import jit, nn, grad, value_and_grad
from jax.lax import scan

from cavi_cmn import Distribution, ArrayDict, Delta
from cavi_cmn.utils import (
    apply_add,
    apply_scale,
    get_default_args,
    check_optim_args,
    bdot,
    inv_and_logdet,
    tree_at,
)
from cavi_cmn import exponential as exp
from cavi_cmn.exponential import Multinomial, MultivariateNormal, MixtureMessage

DEFAULT_EVENT_DIM = 1


class MultinomialRegression(Distribution):

    pytree_data_fields = ("beta", "scale", "prior_inv_sigma_mu")
    pytree_aux_fields = (
        "x_dim",
        "y_dim",
        "use_bias",
        "use_holdout",
        "optim_args",
        "compute_elbo",
    )

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        scale: float = 1.0,
        *,
        batch_shape: tuple = (),
        rng_key: PRNGKeyArray = None,
        use_bias: bool = True,
        use_holdout: bool = False,
        optim_args: Optional[dict] = None,
        compute_elbo: bool = True,
        init_posterior_scale: float = 1e2,
        sample_initial_betas: bool = False,
    ):
        if rng_key is None:
            rng_key = jr.PRNGKey(0)

        self.scale = scale
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.use_bias = use_bias
        self.use_holdout = use_holdout
        optim_args = optim_args if optim_args is not None else get_default_args()
        self.optim_args = check_optim_args(optim_args)
        self.compute_elbo = compute_elbo
        cache_to_compute = "all" if compute_elbo else ["sigma", "mu"]
        K = y_dim - 1 + use_holdout
        d = x_dim + 1
        bias = -jnp.log(K - jnp.arange(K)).reshape(K, 1, 1)
        pad_width = [(0, 0)] * 3
        pad_width[-2] = (d - 1, 0)
        self.prior_inv_sigma_mu = jnp.pad(bias, pad_width) / jnp.square(scale)

        self.beta = self._init_beta(
            x_dim + use_bias,
            K,
            rng_key,
            init_posterior_scale=init_posterior_scale,
            sample_initial_betas=sample_initial_betas,
            batch_shape=batch_shape,
            cache_to_compute=cache_to_compute,
        )

        event_shape = (y_dim,)
        super().__init__(DEFAULT_EVENT_DIM, batch_shape, event_shape)

    def compute_phi(self, X, beta):
        if self.use_bias is False:
            phi = (jnp.expand_dims(X, -3) * beta).sum((-2, -1))
        else:
            phi = (jnp.expand_dims(X, -3) * beta[..., :-1, :]).sum((-2, -1)) + beta[
                ..., -1, -1
            ]

        return phi

    def compute_phiphi(self, X, XX, betabeta):
        if self.use_bias is False:
            phiphi = (jnp.expand_dims(XX, -3) * betabeta).sum((-2, -1))
        else:
            phiphi = (jnp.expand_dims(XX, -3) * betabeta[..., :-1, :-1]).sum((-2, -1))
            phiphi += (jnp.expand_dims(X, -3) * betabeta[..., :-1, -1:]).sum((-2, -1))
            phiphi += (jnp.expand_dims(X.mT, -3) * betabeta[..., -1:, :-1]).sum(
                (-2, -1)
            )
            phiphi += betabeta[..., -1, -1]
        return phiphi

    def compute_N(self, Y):
        n = Y.sum(-1, keepdims=True) - jnp.cumsum(Y, -1) + Y
        if self.use_holdout is True:
            return n
        else:
            return n[..., :-1]

    @staticmethod
    def compute_Ew(pgb, pgc):
        return 0.5 * pgb * jnp.tanh(0.5 * pgc) / pgc

    @staticmethod
    def compute_pg_logZ(pgb, pgc):
        # - pgb * log(cosh(pgc/2))
        return pgb * jnp.log(2.0) + 0.5 * (pgb * pgc) - pgb * nn.softplus(pgc)

    def kl_divergence_qw(self, pgb, pgc):
        return -0.25 * pgb * pgc * jnp.tanh(0.5 * pgc) - self.compute_pg_logZ(pgb, pgc)

    @staticmethod
    def compute_Y_stats(Y):
        if isinstance(Y, Distribution):
            Y = Y.mean
        return Y

    @staticmethod
    def compute_X_stats(X):
        if isinstance(X, Distribution):
            expected_x, expected_xx = X.expected_x(), X.expected_xx()
        else:
            expected_x, expected_xx = X, X * X.mT
        return expected_x, expected_xx

    def ELL_partial(self, Y, X, XX, beta, betabeta, w, weights=None, phiphi=None):
        N = self.compute_N(Y)
        if self.use_holdout is True:
            ymn = Y - 0.5 * N
        else:
            ymn = Y[..., :-1] - 0.5 * N

        phi = self.compute_phi(X, beta)
        phiphi = self.compute_phiphi(X, XX, betabeta) if phiphi is None else phiphi
        if weights is None:
            return (
                (ymn * phi).sum(-1)
                - 0.5 * (w * phiphi).sum(-1)
                - N.sum(-1) * jnp.log(2.0)
            )
        else:
            weights = jnp.expand_dims(weights, -1)
            return (
                (ymn * phi * weights).sum(-1)
                - 0.5 * (w * phiphi * weights).sum(-1)
                - (N * weights).sum(-1) * jnp.log(2.0)
            )

    def ELL(self, Y, X, XX, beta, betabeta, w, *, weights=None):
        return self.ELL_partial(Y, X, XX, beta, betabeta, w, weights=weights).sum()

    @property
    def dELL_dbetas(self):
        return grad(self.ELL, argnums=(3, 4))

    @property
    def dELL_dXs(self):
        return grad(self.ELL, argnums=(1, 2))

    @property
    def dELL_dY(self):
        return grad(self.ELL, argnums=(0,))

    def _init_beta(
        self,
        x_dim: int,
        y_dim: int,
        rng_key: PRNGKeyArray,
        *,
        init_posterior_scale: float,
        cache_to_compute: Optional[List[str]] = None,
        batch_shape: tuple = (),
        sample_initial_betas=False,
    ) -> exp.MultivariateNormal:
        """
        Initialize the parameters of the distribution.
        """
        event_shape = (y_dim, x_dim, 1)

        if (
            cache_to_compute is None
        ):  # defaults to having `self.beta` only comptue `sigma` and `mu` everytime natural parameters are updated
            cache_to_compute = ["sigma", "mu"]

        # inv_sigma = jnp.broadcast_to(jnp.eye(x_dim) * 1e4, batch_shape + (y_dim, x_dim, x_dim))
        inv_sigma = jnp.broadcast_to(
            jnp.eye(x_dim) * init_posterior_scale, batch_shape + (y_dim, x_dim, x_dim)
        )

        if not sample_initial_betas:
            if self.use_bias:
                inv_sigma_mu = inv_sigma @ (self.prior_inv_sigma_mu * self.scale**2)
            else:
                inv_sigma_mu = inv_sigma @ (
                    self.prior_inv_sigma_mu[..., :-1, :] * self.scale**2
                )

            nat_params = ArrayDict(inv_sigma=inv_sigma, inv_sigma_mu=inv_sigma_mu)
        elif sample_initial_betas:
            mu = jr.uniform(
                rng_key, shape=batch_shape + (y_dim, x_dim - 1, 1), minval=-1, maxval=1
            )
            pad_width = [(0, 0)] * mu.ndim
            pad_width[-2] = (0, 1)
            mu = jnp.pad(mu, pad_width=pad_width)
            if self.use_bias:
                inv_sigma_mu = inv_sigma @ (
                    self.prior_inv_sigma_mu * self.scale**2 + mu
                )
            else:  # not sure use_bias=False case is handled correctly / working
                inv_sigma_mu = inv_sigma @ (
                    self.prior_inv_sigma_mu[..., :-1, :] * self.scale**2 + mu
                )

            nat_params = ArrayDict(inv_sigma_mu=inv_sigma_mu, inv_sigma=inv_sigma)

        return exp.MultivariateNormal(
            nat_params=nat_params,
            batch_shape=batch_shape,
            event_shape=event_shape,
            event_dim=len(event_shape),
            cache_to_compute=cache_to_compute,
        )  # computes all the cache elements

    def update_from_probabilities(self, inputs, weights: Optional[Array] = None):
        if isinstance(inputs[0], MixtureMessage):
            inputs = (inputs[0].marginalize_statistics(keepdims=False), inputs[1])
        nat_params, elbo = self._update(
            inputs, weights=weights, compute_elbo=self.compute_elbo, **self.optim_args
        )
        self.beta.nat_params = nat_params
        return elbo

    def update(self, inputs, weights: Optional[Array] = None):
        if isinstance(inputs[0], MixtureMessage):
            inputs = (inputs[0].marginalize_statistics(keepdims=False), inputs[1])
        nat_params, elbo = self._update(
            inputs, weights=weights, compute_elbo=self.compute_elbo, **self.optim_args
        )
        self.beta.nat_params = nat_params
        return elbo

    @staticmethod
    def get_beta_stats(inv_sigma, inv_sigma_mu):
        beta_sigma, beta_inv_sigma_logdet = inv_and_logdet(inv_sigma)
        beta_inv_sigma_logdet = beta_inv_sigma_logdet.squeeze((-2, -1))
        beta_x = bdot(beta_sigma, inv_sigma_mu)
        beta_xx = beta_sigma + beta_x * beta_x.mT

        return beta_x, beta_xx, beta_inv_sigma_logdet

    @partial(jit, static_argnames=["compute_elbo", "iters"])
    def _update(
        self,
        inputs: Union[
            Tuple[Array, Array],
            Tuple[Array, Distribution],
            Tuple[Distribution, Array],
            Tuple[Distribution],
        ],
        weights: Optional[Array] = None,
        compute_elbo: bool = True,
        iters: int = 1,
        lr: float = 1.0,
        beta: Optional[float] = 0.0,
    ):
        """
        Update the parameters of the distribution from the input/output pairs, using multiple iterations of a VB algorithm that works on expectations of auxiliary PG (Polyagamma) variables.
        A more Bayesian, distributional variant of the Jordan & Jakkola trick, where EM-like, fixed-point updates are used to optimize the values of \epsilon parameters, which parameterize
        an approximation to the logistic likelihood in Multinomial Regression models.

        Parameters
        ----------
        data : Tuple[Array] or Tuple[Distribution]
            If Tuple[Array]:
                A tuple of the data, where the first element is `x` and the second element is `y`.
                `X` is (sample_shape, batch_shape, x_dim - 1, 1) if use_bias else (sample_shape, batch_shape, x_dim, 1)
                `y` is (sample_shape, batch_shape, y_dim + 1)
            If Tuple[Distribution]:
                A tuple of incoming probability distributions, where the first element is `x` and the second element is `y`.
                `X` is a distribution of (sample_shape, batch_shape, x_dim - 1, 1) if use_bias else (sample_shape, batch_shape, x_dim, 1)
                `y` is (sample_shape, batch_shape, y_dim + 1)
        """
        pX, pY = inputs
        expected_x, expected_xx = self.compute_X_stats(pX)

        if self.use_bias:
            pad_width = [(0, 0)] * expected_x.ndim
            pad_width[-2] = (0, 1)
            m = jnp.pad(
                expected_x, pad_width=pad_width, mode="constant", constant_values=1.0
            )
            M = jnp.concatenate([expected_xx, expected_x.mT], axis=-2)
            M = jnp.concatenate([M, m], axis=-1)
        else:
            m, M = expected_x, expected_xx

        m, M = jnp.expand_dims(m, -3), jnp.expand_dims(M, -3)

        Y = self.compute_Y_stats(pY)
        sample_dims = tuple(
            range(max(expected_x.ndim - 2, Y.ndim - 1) - self.batch_dim)
        )
        pgb = self.compute_N(Y)

        if self.use_holdout is True:
            kappa = jnp.expand_dims(Y - 0.5 * pgb, (-1, -2))
        else:
            kappa = jnp.expand_dims(Y[..., :-1] - 0.5 * pgb, (-1, -2))

        correction_term = (
            self.prior_inv_sigma_mu[:, :-1, :]
            if not self.use_bias
            else self.prior_inv_sigma_mu
        )
        beta_prior_params = ArrayDict(
            inv_sigma_mu=(1 - beta) * correction_term + self.beta.inv_sigma_mu * beta,
            inv_sigma=self.beta.inv_sigma * beta
            + (1.0 - beta)
            * jnp.eye(self.x_dim + self.use_bias)
            / jnp.square(self.scale),
        )

        def vb_iter_step(carry, _):
            beta_inv_sigma_mu, beta_inv_sigma = carry
            beta_x, beta_xx, beta_inv_sigma_logdet = self.get_beta_stats(
                beta_inv_sigma, beta_inv_sigma_mu
            )

            phiphi = self.compute_phiphi(expected_x, expected_xx, beta_xx)
            pgc = jnp.sqrt(phiphi)
            expected_w = self.compute_Ew(pgb, pgc)

            beta_inv_sigma_mu = beta_prior_params.inv_sigma_mu + jnp.sum(
                kappa * m, axis=sample_dims
            )
            beta_inv_sigma = beta_prior_params.inv_sigma + jnp.sum(
                jnp.expand_dims(expected_w, (-1, -2)) * M, axis=sample_dims
            )

            if compute_elbo:
                beta_x, beta_xx, beta_inv_sigma_logdet = self.get_beta_stats(
                    beta_inv_sigma, beta_inv_sigma_mu
                )
                elbo = self.ELL_partial(
                    Y, expected_x, expected_xx, beta_x, beta_xx, expected_w, weights
                ).sum(sample_dims)
                elbo -= self.kl_divergence_qw(pgb, pgc).sum(sample_dims + (-1,))
                elbo -= self.kl_divergence_fx(beta_xx, beta_inv_sigma_logdet)
                # TODO: elbo computation is missing D_kl(q(x|z)||p(x))
            else:
                elbo = None

            return (beta_inv_sigma_mu, beta_inv_sigma), elbo

        beta_inv_sigma_mu = self.beta.inv_sigma_mu
        beta_inv_sigma = self.beta.inv_sigma
        (beta_inv_sigma_mu, beta_inv_sigma), elbo = scan(
            vb_iter_step, (beta_inv_sigma_mu, beta_inv_sigma), jnp.arange(iters)
        )

        beta_params = ArrayDict(
            inv_sigma_mu=beta_inv_sigma_mu, inv_sigma=beta_inv_sigma
        )
        new_beta_params = apply_add(
            apply_scale(beta_prior_params, 1.0 - lr), apply_scale(beta_params, lr)
        )
        return new_beta_params, elbo

    def kl_divergence_fx(
        self, beta_xx, logdet_inv_sigma
    ):  # Assumes prior on betas is Normal(0, sigma^2)
        x_dim = beta_xx.shape[-2]
        y_dim = beta_xx.shape[-3]
        KL = x_dim * y_dim * jnp.log(self.scale) + 0.5 * logdet_inv_sigma.sum(-1)
        KL = KL - 0.5 * x_dim * y_dim
        KL = KL + 0.5 * jnp.trace(beta_xx, axis1=-1, axis2=-2).sum(-1) / jnp.square(
            self.scale
        )
        return KL

    def kl_divergence(self):
        # beta dims
        scale_sqr = jnp.square(self.scale)
        x_dim = self.x_dim + self.use_bias
        y_dim = self.y_dim - 1
        KL = x_dim * y_dim * jnp.log(self.scale) - 0.5 * x_dim * y_dim
        KL += 0.5 * self.beta.compute_logdet_inv_sigma().sum((-3, -2, -1))
        if self.use_bias:
            mu = self.prior_inv_sigma_mu * scale_sqr - self.beta.mean
        else:
            mu = self.prior_inv_sigma_mu[..., :-1, :] * scale_sqr - self.beta.mean

        KL += 0.5 * jnp.trace(
            (self.beta.sigma + mu * mu.mT) / scale_sqr, axis1=-1, axis2=-2
        ).sum(-1)

        return KL

    def log_predict(self, X):
        return self.log_forward(Delta(values=X, event_dim=2))

    def predict(self, X):
        return self.forward(Delta(values=X, event_dim=2))

    @multimethod
    def log_forward(self, pX: MixtureMessage, iters=4) -> Array:
        # pX_marg = pX.marginalize_statistics(keepdims=False)
        # pX_delta = Delta(values=pX_marg.mu, event_dim=2)
        # return self.log_forward(pX_delta)
        # pX_marg = pX.marginalize_statistics(keepdims=False)
        # return self.log_forward(pX_marg, iters=iters)

        # compute \log p(y|z) = \log \int p(y|x) p(x|z) dx
        # should have shape (n_samples, *batch_shape, z_dim, y_dim)
        log_py_z = jnp.moveaxis(self.log_forward(pX.likelihood.moveaxis(-1, 0)), 0, -2)

        # normalizes across y_dim (the event_shape of the MNLR)
        log_py_z = log_py_z - logsumexp(log_py_z, -1, keepdims=True)

        # compute \log p(z)
        # should have shape (n_samples, *batch_shape, z_dim)
        log_pz = pX.assignments.logits + pX.likelihood.residual
        log_pz = log_pz - logsumexp(
            log_pz, -1, keepdims=True
        )  # normalizes across z_dim

        # compute \log p(y) = \log (\int p(y|z) p(z) dz)
        # jnp.expand_dims(log_pz, -1) has shape (n_samples, *batch_shape, z_dim, 1)
        # log_py_z + jnp.expand_dims(log_pz, -1) has shape (n_samples, n_models, z_dim, y_dim)
        # logsumexp(..., -2) sums over z_dim, resulting in shape (n_samples, n_models, y_dim)
        logits = logsumexp(log_py_z + jnp.expand_dims(log_pz, -1), -2)
        return logits

    @multimethod
    def log_forward(self, pX: Distribution, iters=4) -> Array:
        return self.joint(pX, iters=iters)[1]

    @multimethod
    def log_forward(self, pX: Delta) -> Array:

        sample_n_batch_len = len(pX.shape[:-2])
        Y = jnp.expand_dims(
            jnp.eye(self.y_dim), tuple(range(1, sample_n_batch_len + 1))
        )

        beta_x, beta_xx = self.compute_X_stats(self.beta)
        px_x, px_xx = self.compute_X_stats(pX)

        pgb = self.compute_N(Y)

        if self.use_holdout is True:
            kappa = Y - 0.5 * pgb
        else:
            kappa = Y[..., :-1] - 0.5 * pgb

        pgc = jnp.sqrt(self.compute_phiphi(px_x, px_xx, beta_xx))
        phi = self.compute_phi(px_x, beta_x)

        res = jnp.sum(kappa * phi + 0.5 * (pgb * pgc) - pgb * nn.softplus(pgc), -1)
        logits = jnp.moveaxis(res, 0, -1)

        return logits

    @multimethod
    def forward(self, pX: Delta) -> Multinomial:
        logits = self.log_forward(pX)
        res = logsumexp(logits, -1, keepdims=True)
        logits = logits - res
        return Multinomial(ArrayDict(logits=logits), residual=res.squeeze(-1))

    @multimethod
    def forward(self, pX: Distribution, iters=4) -> Multinomial:
        logits = self.log_forward(pX, iters=iters)
        res = logsumexp(logits, -1, keepdims=True)
        logits = logits - res
        return Multinomial(ArrayDict(logits=logits), residual=res.squeeze(-1))

    def joint(self, pX: MultivariateNormal, iters=4, categorical_axis=False):

        sample_n_batch_len = len(pX.shape[:-2]) - int(categorical_axis)
        Y = jnp.expand_dims(
            jnp.eye(self.y_dim), tuple(range(1, sample_n_batch_len + 1))
        )

        beta_x, beta_xx = self.compute_X_stats(self.beta)
        px_x, px_xx = self.compute_X_stats(pX)

        shape = jnp.broadcast_shapes(pX.shape[:-2] + (1,), self.beta.shape[:-2])
        pgc = 1e-6 * jnp.ones(shape)
        pgb, pgc = jnp.broadcast_arrays(self.compute_N(Y), pgc)

        qX, _ = self.__backward_smooth(
            Y, (pX, px_x, px_xx), (beta_x, beta_xx), iters=iters, params=(pgb, pgc)
        )

        inv_sigma_mu = jnp.moveaxis(qX.inv_sigma_mu, 0, -3)
        inv_sigma = jnp.moveaxis(qX.inv_sigma, 0, -3)
        logits = jnp.moveaxis(qX.residual, 0, -1)
        qX = MultivariateNormal(
            ArrayDict(inv_sigma_mu=inv_sigma_mu, inv_sigma=inv_sigma)
        )
        return qX, logits

    def backward(self, Y, params=None, iters=8):
        # This routine uses a pgc presumably computed from a previous forward pass to compute Ew which is then
        # used to compute the expected natural parameters of the likelihood of X
        if isinstance(Y, Distribution):
            Y = Y.mean

        shape = Y.shape[:-1]
        qX = MultivariateNormal(
            ArrayDict(
                inv_sigma_mu=jnp.zeros(shape + (self.x_dim, 1)),
                inv_sigma=jnp.broadcast_to(
                    jnp.eye(self.x_dim), shape + (self.x_dim, self.x_dim)
                ),
            )
        )

        if params is None:
            shape = jnp.broadcast_shapes(qX.shape[:-2] + (1,), self.beta.shape[:-2])
            pgc = 1e-6 * jnp.ones(shape)
            pgb, pgc = jnp.broadcast_arrays(self.compute_N(Y), pgc)
            params = (pgb, pgc)

        beta_x, beta_xx = self.compute_X_stats(self.beta)
        px_x, px_xx = self.compute_X_stats(qX)

        qx, (pgb, pgc) = self.__backward_smooth(
            Y, (qX, px_x, px_xx), (beta_x, beta_xx), iters=iters, params=params
        )

        return qx

    def __backward_smooth(self, y_stats, x_stats, beta_stats, *, iters, params=None):

        px, expected_x, expected_xx = x_stats
        beta_x, beta_xx = beta_stats

        sample_shape = jnp.broadcast_shapes(
            expected_x.shape[: -2 - self.batch_dim],
            y_stats.shape[: -1 - self.batch_dim],
        )
        inv_sigma_mu = jnp.zeros(sample_shape + self.batch_shape + (self.x_dim, 1))
        inv_sigma = jnp.broadcast_to(
            jnp.eye(self.x_dim),
            sample_shape + self.batch_shape + (self.x_dim, self.x_dim),
        )
        exp_x = jnp.zeros_like(inv_sigma_mu)
        exp_xx = jnp.zeros_like(inv_sigma)

        if params is None:
            pgb = self.compute_N(y_stats)
            pgc = jnp.sqrt(self.compute_phiphi(expected_x, expected_xx, beta_xx))
            pgb, pgc = jnp.broadcast_arrays(pgb, pgc)
        else:
            pgb, pgc = params

        if self.use_holdout is True:
            kappa = y_stats - 0.5 * pgb
        else:
            kappa = y_stats[..., :-1] - 0.5 * pgb

        def step_fn(carry, _):
            pgc, _, _, _, _ = carry

            # update q(w)
            expected_w = jnp.expand_dims(self.compute_Ew(pgb, pgc), (-1, -2))

            # update q(x)
            inv_sigma_mu = px.inv_sigma_mu

            if self.use_bias:
                inv_sigma_mu += jnp.sum(
                    jnp.expand_dims(kappa, (-1, -2)) * beta_x[..., :-1, :], -3
                )
                m = (beta_xx[..., :-1, -1:] + beta_xx[..., -1:, :-1].mT) / 2
                inv_sigma_mu -= jnp.sum(expected_w * m, -3)
            else:
                inv_sigma_mu += jnp.sum(jnp.expand_dims(kappa, (-1, -2)) * beta_x, -3)

            inv_sigma = px.inv_sigma
            if self.use_bias:
                inv_sigma += jnp.sum(expected_w * beta_xx[..., :-1, :-1], -3)
            else:
                inv_sigma += jnp.sum(expected_w * beta_xx, -3)

            sigma = inv_and_logdet(inv_sigma, return_logdet=False)
            expected_x = bdot(sigma, inv_sigma_mu)
            expected_xx = sigma + expected_x * expected_x.mT

            pgc = jnp.sqrt(self.compute_phiphi(expected_x, expected_xx, beta_xx))

            return (pgc, inv_sigma, inv_sigma_mu, expected_x, expected_xx), None

        (pgc, inv_sigma, inv_sigma_mu, exp_x, exp_xx), _ = scan(
            step_fn,
            (pgc, inv_sigma, inv_sigma_mu, exp_x, exp_xx),
            jnp.arange(iters),
            unroll=2,
        )

        qx = MultivariateNormal(
            expectations=ArrayDict(x=exp_x, minus_half_xxT=-exp_xx / 2),
            cache_to_compute=["mu", "sigma"],
        )
        qx_logdet_inv_sigma = inv_and_logdet(inv_sigma, return_inverse=False).squeeze(
            (-1, -2)
        )
        # exp_x, _ = self.compute_X_stats(qx)
        phi = self.compute_phi(exp_x, beta_x)
        res = jnp.sum(kappa * phi + 0.5 * (pgb * pgc) - pgb * nn.softplus(pgc), -1)

        # negative KL diverge q(x) , p(x) for residuals
        _mu = px.mean - qx.mean
        kl_div = jnp.sum(px.inv_sigma * (qx.sigma + _mu * _mu.mT), axis=(-1, -2)) / 2
        kl_div += (qx_logdet_inv_sigma - px.logdet_inv_sigma.squeeze((-1, -2))) / 2
        kl_div -= qx.dim / 2
        res -= kl_div

        qx.residual = res

        return qx, (pgb, pgc)

    def backward_smooth(self, pY: Distribution, pX: MultivariateNormal, iters=4):
        # New backward_smooth method, that computes the posterior
        # over pX given pY and the current q(beta). Similar to backward_smooth below
        # but actually returns the exact posterior over q(x_{l-1})

        Y = self.compute_Y_stats(pY)

        beta_x, beta_xx = self.compute_X_stats(self.beta)
        px_x, px_xx = self.compute_X_stats(pX)

        shape = jnp.broadcast_shapes(pX.shape[:-2] + (1,), self.beta.shape[:-2])
        pgc = 1e-6 * jnp.ones(shape)
        pgb, pgc = jnp.broadcast_arrays(self.compute_N(Y), pgc)

        qX, (pgb, pgc) = self.__backward_smooth(
            Y, (pX, px_x, px_xx), (beta_x, beta_xx), iters=iters, params=(pgb, pgc)
        )

        return qX

    def expected_log_likelihood(self, X, Y, pg_params=None):
        return self.expected_log_likelihood_from_ELL_partial(X, Y, pg_params=pg_params)

    def expected_log_likelihood_from_forward(self, pX, pY):
        # This version computes the expected log likelihood by averaging out q(x) and q(beta) in the
        # log domain and then marginalizes out the polyagamma variable eactly
        if isinstance(pY, Distribution):
            Y = pY.mean
        else:
            Y = pY
        logits, pgc = self.log_forward(pX)

        pgb = self.compute_N(Y)
        if self.use_holdout == False:
            ell = (logits * Y).sum(-1)
        else:
            ell = (logits[..., :-1] * Y).sum(-1)
        return ell, (pgb, pgc)

    def expected_log_likelihood_from_ELL_partial(self, X, Y, pg_params=None):
        Y = self.compute_Y_stats(Y)
        X, XX = self.compute_X_stats(X)
        beta_x, beta_xx = self.compute_X_stats(self.beta)
        if pg_params is None:
            pgb = self.compute_N(Y)
            phiphi = self.compute_phiphi(X, XX, beta_xx)
            pgc = jnp.sqrt(phiphi)
        else:
            pgb, pgc = pg_params

        exp_w = self.compute_Ew(pgb, pgc)

        ell = self.ELL_partial(Y, X, XX, beta_x, beta_xx, exp_w, phiphi=phiphi)
        ell -= self.kl_divergence_qw(pgb, pgc).sum(-1)

        return ell, (pgb, pgc)

    def expected_log_likelihood_from_backward(self, pY, pX):
        raise NotImplementedError

    def elbo_contrib(self, X, Y, weights=None):
        sample_dims = tuple(range(max(X.ndim - 2, Y.ndim - 1) - self.batch_dim))
        ell = self.expected_log_likelihood(X, Y)[0]
        if weights is None:
            ell = ell.sum(sample_dims)
        else:
            ell = (ell * weights).sum(sample_dims)
        ell -= self.kl_divergence()
        return ell

    def W_hat(self):
        if self.use_holdout == False:
            return 2 * self.beta.mean - self.beta.mean.cumsum(-3)
        else:
            return (2 * self.beta.mean - self.beta.mean.cumsum(-3))[..., :-1, :, :]
