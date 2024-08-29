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

from typing import Optional, Tuple
from jaxtyping import Array

from jax import numpy as jnp
from jax import random as jr
from jax import tree_util
from jax import scipy as jsp

from cavi_cmn.distribution import Distribution
from cavi_cmn.exponential import Linear as LinearLikelihood
from cavi_cmn.exponential import MultivariateNormal, ExponentialFamily
from cavi_cmn.utils import (
    mvdigamma,
    mvgammaln,
    params_to_tx,
    ArrayDict,
    bdot,
    tree_at,
    inv_and_logdet,
)
from cavi_cmn.transforms import Transform

DEFAULT_EVENT_DIM = 2


@params_to_tx({"eta_1": "xx", "eta_2": "yx", "eta_3": "yy", "eta_4": "ones"})
class Linear(Transform):
    """
    This distribution models the linear transformation:

    .. math::
       y = Ax + \\epsilon

    where:
        :math:`y` is the output vector with size p
        :math:`x` is the input vector with size d
        :math:`A` are the linear transformation parameters with size p x d
        :math:`\\epsilon` is additive Gaussian noise

    The distribution models this transformation with the following likelihood:

    .. math::
        \\log p(y|x,A,\\Sigma^-1) = - 0.5 * (y - Ax)^T \\Sigma^{-1}(y - Ax)
                               + 0.5 \\log |\\Sigma^{-1}| - \\frac{d}{2} \\log (2 \\pi)

    where:
        :math: `Sigma` is a positive definite covariance matrix with size p x p

    The conjugate prior to :math:`\\A` and :math:`\\Sigma` is the
    Matrix Normal Wishart distribution:

    .. math::
        A | \\Sigma ~ \\mathcal{MN}(A | \\mu_0, \\Sigma, V_0)
        \\Sigma^{-1}     ~ \\mathcal{W}(\\Sigma^{-1} | U_0, n_0)

    where:
        :math:`\\mu_0` is the prior mean of the linear transformation
        :math:`V_0` is the prior covariance of the linear transformation
        :math:`U_0` is the prior scale matrix of the precision matrix
        :math:`n_0` is the prior degrees of freedom of the precision matrix

    The full joint distribution is then given by:

    .. math::
        log p(y,x,A,\\Sigma) = log p(y|x, A, \\Sigma) + log p(A | \\Sigma) + log p(\\Sigma)

    The posterior distribution is also a Matrix Normal Wishart distribution:

    .. math::
        Sigma = L.T L
        V = B.T B
        A = M + L Z B.T
        y = A x + L = L ( L^-1 A x + I) =
        A | \\Sigma ~ \\mathcal{MN}(A | \\mu, \\Sigma, V)
        \\Sigma^{-1}     ~ \\mathcal{W}(\\Sigma^{-1} | U, n)

    where:
        :math:`\\mu` is the posterior mean of the linear transformation
        :math:`V` is the posterior covariance of the linear transformation
        :math:`U` is the posterior scale matrix of the precision matrix
        :math:`n` is the posterior degrees of freedom of the precision matrix

    This distribution utilises the following (natural) parameterisation:

    .. math::
        \\eta_1 = V^{-1}
        \\eta_2 = V^{-1} \\mu
        \\eta_3 = U^{-1} + \\mu V^{-1} \\mu^T
        \\eta_4 = n + d - p - 1

    which have the following sufficient statistics T(x):

    .. math::
        \\eta_1 = \\sum_{i=1}^n x_i x_i^T
        \\eta_2 = \\sum_{i=1}^n y_i x_i^T
        \\eta_3 = \\sum_{i=1}^n y_i y_i^T
        \\eta_4 = n

    """

    _u: Array
    _v: Array
    _logdet_inv_v: Array
    _logdet_inv_u: Array
    _prior_logdet_inv_v: Array
    _prior_logdet_inv_u: Array

    x_dim: int
    y_dim: int

    pytree_data_fields = (
        "_u",
        "_v",
        "_logdet_inv_v",
        "_logdet_inv_u",
        "_prior_logdet_inv_v",
        "_prior_logdet_inv_u",
        "_prior_v",
    )
    pytree_aux_fields = (
        "x_dim",
        "y_dim",
        "use_bias",
        "fixed_precision",
        "trivial_batch_axes",
    )

    def __init__(
        self,
        params: ArrayDict = None,
        prior_params: ArrayDict = None,
        event_dim: int = DEFAULT_EVENT_DIM,
        use_bias: bool = True,
        fixed_precision: bool = False,
        scale: float = 1.0,
        dof_offset: float = 0.0,  # offset for the prior degrees of freedom (n = 1 + dim + dof_offset) If dof_offset = 0.0 then expected_sigma() is undefined, so one can use dof_offset=1.0 if expected_sigma() is needed
        inv_v_scale: float = 1.0,  # scale for the prior precision matrix V^{-1} of the linear transformation
        batch_shape: Optional[Tuple[int]] = None,
        event_shape: Optional[Tuple[int]] = None,
        init_key=None,
    ):

        if params is not None:
            self.x_dim, self.y_dim = (
                params.mu.shape[-(DEFAULT_EVENT_DIM - 1)],
                params.mu.shape[-DEFAULT_EVENT_DIM],
            )

        elif prior_params is not None:
            self.x_dim, self.y_dim = (
                prior_params.mu.shape[-(DEFAULT_EVENT_DIM - 1)],
                prior_params.mu.shape[-DEFAULT_EVENT_DIM],
            )

        elif event_shape is not None:
            self.x_dim, self.y_dim = (
                event_shape[-(DEFAULT_EVENT_DIM - 1)],
                event_shape[-DEFAULT_EVENT_DIM],
            )

        self.use_bias = use_bias
        self.fixed_precision = fixed_precision

        if prior_params is None:
            prior_params = self.init_default_params(
                batch_shape,
                event_shape,
                self.x_dim,
                self.y_dim,
                scale,
                dof_offset,
                inv_v_scale,
                DEFAULT_EVENT_DIM,
            )

        if (
            fixed_precision
        ):  # this is so that _update_cache() works when `super().__init__()` is called below
            self._prior_v = jnp.linalg.inv(prior_params.inv_v)

        if params is None:
            key = init_key if init_key is not None else jr.PRNGKey(0)
            mu = prior_params.mu + jr.uniform(
                key,
                batch_shape + event_shape[:-event_dim] + (self.y_dim, self.x_dim),
                minval=-2 * scale / jnp.sqrt(inv_v_scale),
                maxval=2 * scale / jnp.sqrt(inv_v_scale),
            )

            inv_v = jnp.where(prior_params.inv_v > 0.0, 1.0, 0.0)
            inv_u = jnp.where(prior_params.inv_u > 0.0, 0.5, 0.0)
            params = tree_at(
                lambda x: (x.mu, x.inv_v, x.inv_u), prior_params, (mu, inv_v, inv_u)
            )

        inferred_batch_shape, inferred_event_shape = self.infer_shapes(
            params.mu, event_dim
        )
        if batch_shape is None:
            batch_shape = inferred_batch_shape
        if event_shape is None:
            event_shape = inferred_event_shape

        super().__init__(
            DEFAULT_EVENT_DIM,
            LinearLikelihood,
            params,
            prior_params,
            batch_shape,
            event_shape,
        )

        self._prior_v, self._prior_logdet_inv_v = inv_and_logdet(self.prior_inv_v)
        self._prior_logdet_inv_u = inv_and_logdet(
            self.prior_inv_u, return_inverse=False
        )
        self.trivial_batch_axes = tuple(
            [i for i, d in enumerate(batch_shape) if d == 1]
        )

    @staticmethod
    def init_default_params(
        batch_shape,
        event_shape,
        x_dim,
        y_dim,
        scale=1.0,
        dof_offset=0.0,
        inv_v_scale=1.0,
        default_event_dim=2,
    ) -> ArrayDict:
        """
        Initialize the default parameters of the distribution.
        """

        prior_mu = jnp.full(
            batch_shape + event_shape[:-default_event_dim] + (y_dim, x_dim), 0.0
        )  # shape should be (batch_shape,) + (y_dim, x_dim)
        prior_n = jnp.full(
            batch_shape + event_shape[:-default_event_dim] + (1, 1),
            1.0
            + y_dim
            + dof_offset,  # NOTE: defining n = 1.0 + dim means expected_sigma() is undefined. In the past, Jeff has used n = 2.0 + dim to address this
        )  # shape should be (batch_shape,) + (1, 1)
        prior_inv_v = inv_v_scale * jnp.broadcast_to(
            jnp.eye(x_dim),
            batch_shape + event_shape[:-default_event_dim] + (x_dim, x_dim),
        )  # shape should be (batch_shape,) + (x_dim, x_dim).
        prior_inv_u = scale**2 * jnp.broadcast_to(
            jnp.eye(y_dim),
            batch_shape + event_shape[:-default_event_dim] + (y_dim, y_dim),
        )  # shape should be (batch_shape,) + (y_dim, y_dim).

        return ArrayDict(mu=prior_mu, inv_v=prior_inv_v, inv_u=prior_inv_u, n=prior_n)

    @property
    def mu(self):
        return bdot(self.posterior_params.eta.eta_2, self.v)

    @property
    def inv_v(self):
        return self.posterior_params.eta.eta_1

    @property
    def inv_u(self):
        if self.fixed_precision:
            return self.prior_inv_u
        else:
            return self.posterior_params.eta.eta_3 - bdot(
                self.mu, bdot(self.inv_v, self.mu.mT)
            )

    @property
    def n(self):
        if self.fixed_precision:
            return self.prior_n
        else:
            return self.posterior_params.eta.eta_4 + self.y_dim - self.x_dim + 1

    @property
    def prior_mu(self):
        return bdot(self.prior_params.eta.eta_2, self.prior_v)

    @property
    def prior_inv_v(self):
        return self.prior_params.eta.eta_1

    @property
    def prior_inv_u(self):
        return self.prior_params.eta.eta_3 - bdot(
            self.prior_mu, bdot(self.prior_inv_v, self.prior_mu.mT)
        )

    @property
    def prior_n(self):
        return self.prior_params.eta.eta_4 + self.y_dim - self.x_dim + 1

    @property
    def v(self):
        if self._v is None:
            self._v, self._logdet_inv_v = inv_and_logdet(self.inv_v)
        return self._v

    @property
    def prior_v(self):
        if self._prior_v is None:
            self._prior_v, self._prior_logdet_inv_v = inv_and_logdet(self.prior_inv_v)
        return self._prior_v

    @property
    def u(self):
        if self._u is None:
            self._u, self._logdet_inv_u = inv_and_logdet(self.inv_u)
        return self._u

    @property
    def logdet_inv_v(self):
        if self._logdet_inv_v is None:
            self._logdet_inv_v = inv_and_logdet(self.inv_v, return_inverse=False)
        return self._logdet_inv_v

    @property
    def logdet_inv_u(self):
        if self._logdet_inv_u is None:
            self._logdet_inv_u = inv_and_logdet(self.inv_u, return_inverse=False)
        return self._logdet_inv_u

    @property
    def prior_logdet_inv_v(self):
        if self._prior_logdet_inv_v is None:
            self._prior_logdet_inv_v = inv_and_logdet(
                self.prior_inv_v, return_inverse=False
            )
        return self._prior_logdet_inv_v

    @property
    def prior_logdet_inv_u(self):
        if self._prior_logdet_inv_u is None:
            self._prior_logdet_inv_u = inv_and_logdet(
                self.prior_inv_u, return_inverse=False
            )
        return self._prior_logdet_inv_u

    @property
    def weights(self):
        return self.mu[..., :-1] if self.use_bias else self.mu

    @property
    def bias(self):
        return (
            self.mu[..., -1:]
            if self.use_bias
            else jnp.broadcast_to(jnp.zeros(1), self.mu.shape[:-1] + (1,))
        )

    def update_from_probabilities(
        self,
        pXY: Tuple[Distribution],
        weights: Optional[Array] = None,
        lr: float = 1.0,
        beta: float = 0.0,
        apply_updates: bool = True,
    ):
        """
        Custom version of `update_from_probs` that accelerates computation
        in case of self.use_bias == True by reducing number of calls to `jnp.concatenate`
        """
        pX, pY = pXY
        sample_shape = self.get_sample_shape(pX.mean)
        sample_dims = self.get_sample_dims(pX.mean)

        pX_batch_shape = self.get_batch_shape(pX.mean)
        pY_batch_shape = self.get_batch_shape(pY.mean)

        common_batch_shape = jnp.broadcast_shapes(
            pX_batch_shape, pY_batch_shape
        )  # this is the batch_shape of the input distributions

        if weights is None:
            num_samples = jnp.prod(jnp.array(sample_shape))
            xx = pX.expected_xx().sum(sample_dims)
            yx = (pY.expected_x() * pX.expected_x().mT).sum(sample_dims)
            yy = pY.expected_xx().sum(sample_dims)
            ones = jnp.broadcast_to(num_samples, xx.shape[:-2] + (1, 1))
            summed_stats = ArrayDict(xx=xx, yx=yx, yy=yy, ones=ones)
        else:
            weights = self.expand_event_dims(weights)
            weights_batch_shape = self.get_batch_shape(weights)
            common_batch_shape = jnp.broadcast_shapes(
                common_batch_shape, weights_batch_shape
            )
            summed_stats = ArrayDict(
                xx=(pX.expected_xx() * weights).sum(sample_dims),
                yx=(pY.expected_x() * pX.expected_x().mT * weights).sum(sample_dims),
                yy=(pY.expected_xx() * weights).sum(sample_dims),
                ones=(weights).sum(sample_dims),
            )

        if self.use_bias:
            if weights is None:
                SEx = pX.mean.sum(sample_dims)
                SEy = pY.mean.sum(sample_dims)
            else:
                SEx = (pX.mean * weights).sum(sample_dims)
                SEy = (pY.mean * weights).sum(sample_dims)

            SExx = jnp.concatenate(
                (summed_stats.xx, SEx), axis=-1
            )  # shape should be batch_shape + event_shape[:-2] + (x_dim, x_dim+1)
            SEx = jnp.concatenate(
                (SEx, summed_stats.ones), axis=-2
            )  # shape should be batch_shape + event_shape[:-2] + (x_dim+1, 1)

            summed_stats = tree_at(
                lambda x: (x.xx, x.yx),
                summed_stats,
                (
                    jnp.concatenate(
                        [SExx, SEx.mT], axis=-2
                    ),  # shape should be batch_shape + event_shape[:-2] + (x_dim+1, x_dim+1)
                    jnp.concatenate(
                        [
                            summed_stats.yx,
                            jnp.broadcast_to(
                                SEy, common_batch_shape + SEy.shape[(-pY.event_dim) :]
                            ),
                        ],
                        axis=-1,
                    ),  # shape should be batch_shape + event_shape[:-2] + (y_dim, x_dim+1)
                ),
            )

        # TODO: do the summation when updating parameters. No need for an additional for loop.
        summed_stats = tree_util.tree_map(
            lambda se: se.sum(self.trivial_batch_axes, keepdims=True), summed_stats
        )
        if apply_updates:
            self.update_from_statistics(
                self.map_stats_to_params(summed_stats, None), lr, beta
            )
        else:
            return self.map_stats_to_params(summed_stats, None)

    def update_from_data(
        self,
        data: Tuple[Array],
        weights: Optional[Array] = None,
        lr: float = 1.0,
        beta: float = 0.0,
    ):
        """
        Custom version of `update_from_data` that accelerates computation in case of self.use_bias == True
        by reducing number of calls to `jnp.concatenate`
        """

        likelihood_stats = self.likelihood.statistics(
            data
        )  # evaluate the sufficient statistics T(x)

        X, Y = data
        sample_dims = self.get_sample_dims(X)

        if weights is None:
            summed_stats = tree_util.tree_map(
                lambda x: x.sum(sample_dims),
                likelihood_stats,
                is_leaf=lambda x: isinstance(x, Array),
            )
        else:
            weights = weights.reshape(weights.shape + self.event_dim * (1,))
            summed_stats = tree_util.tree_map(
                lambda x: (x * weights).sum(sample_dims),
                likelihood_stats,
                is_leaf=lambda x: isinstance(x, Array),
            )

        if self.use_bias:
            if weights is None:
                SEx = X.sum(sample_dims)
                SEy = Y.sum(sample_dims)
            else:
                SEx = (X * weights).sum(sample_dims)
                SEy = (Y * weights).sum(sample_dims)

            SExx = jnp.concatenate(
                (summed_stats.xx, SEx), axis=-1
            )  # shape should be batch_shape + event_shape[:-2] + (x_dim, x_dim+1)
            #            SEx = jnp.concatenate((SEx, jnp.broadcast_to(N[...,None,None],SEx.shape[:-2]+(1,1))), axis=-2) # shape should be batch_shape + event_shape[:-2] + (x_dim+1, 1)
            SEx = jnp.concatenate(
                (SEx, summed_stats.ones), axis=-2
            )  # shape should be batch_shape + event_shape[:-2] + (x_dim+1, 1)

            summed_stats = tree_at(
                lambda x: (x.xx, x.yx),
                summed_stats,
                (
                    jnp.concatenate(
                        (SExx, SEx.mT), axis=-2
                    ),  # shape should be batch_shape + event_shape[:-2] + (x_dim+1, x_dim+1)
                    jnp.concatenate(
                        (summed_stats.yx, SEy), axis=-1
                    ),  # shape should be batch_shape + event_shape[:-2] + (y_dim, x_dim+1)
                ),
            )

        self.update_from_statistics(
            self.map_stats_to_params(summed_stats, None), lr, beta
        )

    def to_natural_params(self, params: ArrayDict) -> ArrayDict:
        """
        Convert the given parameters to natural parameters.

        .. math::
            \\eta_1 = V^{-1}
            \\eta_2 = V^{-1} \\mu
            \\eta_3 = U^{-1} + \\mu V^{-1} \\mu^T
            \\eta_4 = n - d + p - 1
        """
        eta_1 = params.inv_v
        eta_2 = bdot(params.mu, params.inv_v)
        eta_3 = params.inv_u + bdot(params.mu, bdot(params.inv_v, params.mu.mT))
        eta_4 = params.n - self.y_dim + self.x_dim - 1
        return ArrayDict(
            eta=ArrayDict(eta_1=eta_1, eta_2=eta_2, eta_3=eta_3, eta_4=eta_4), nu=None
        )

    def expected_posterior_statistics(self) -> ArrayDict:
        """
        Computes the expected sufficient statistics of the posterior distribution's parameters η and v.

        .. math::
            (<S(θ)>_{q(θ|η, v)}, -<A(θ)>_{q(θ|η, v)})

        N.B. the negation before the return <A(θ)>_{q(θ|η, v)} is done in order
        to make it ready for computing the dot products that help with things like `expected_log_likelihood`
        """
        eta_stats = self.expected_likelihood_params()
        return ArrayDict(eta=eta_stats, nu=None)

    def expected_likelihood_params(self) -> ArrayDict:
        """
        Compute the expected likelihood parameters.
        """

        exp_inv_sigma = self.expected_inv_sigma()
        exp_inv_sigma_x = self.expected_inv_sigma_x()
        exp_x_inv_sigma_x = self.y_dim * self.v + bdot(self.mu.mT, exp_inv_sigma_x)

        return ArrayDict(
            eta_1=-0.5 * exp_x_inv_sigma_x,
            eta_2=exp_inv_sigma_x,
            eta_3=-0.5 * exp_inv_sigma,
            eta_4=0.5 * self.expected_logdet_inv_sigma(),
        )

    def expected_log_partition(self):
        """
        Computes the log partition A(θ) of the likelihood expected under the variational distribution,
        i.e., <A(θ)>_{q(θ|η, v)}
        """
        return 0.5 * (self.expected_x_inv_sigma_x() - self.expected_logdet_inv_sigma())

    def log_prior_partition(self) -> Array:
        """
        Computes the log partition function of the prior distribution, log Z(η₀, ν₀).
        """
        return self._log_partition(
            self.prior_mu,
            self.prior_logdet_inv_v,
            self.prior_logdet_inv_u,
            self.prior_n,
        )

    def log_posterior_partition(self) -> Array:
        """
        Computes the log partition function of the posterior distribution, log Z(η, v).
        """
        return self._log_partition(
            self.mu, self.logdet_inv_v, self.logdet_inv_u, self.n
        )

    def _log_partition(
        self, mean: Array, logdet_inv_v: Array, logdet_inv_u: Array, n: Array
    ) -> Array:
        d = self.y_dim
        p = self.x_dim

        term_1 = ((d * p) / 2) * jnp.log(2 * jnp.pi)
        term_2 = -(d / 2) * logdet_inv_v
        term_3 = ((d * n) / 2) * jnp.log(2)
        term_4 = -(n / 2) * logdet_inv_u
        term_5 = mvgammaln(n / 2.0, d)

        return term_1 + term_2 + term_3 + term_4 + term_5

    def predict(self, x: Array) -> ExponentialFamily:
        """
        Computes the variational prediction distribution.
        @TODO: Adapt to two settings of self.use_bias
        """

        if self.use_bias:
            inv_sigma_mu = (
                bdot(self.expected_inv_sigma_x()[..., :-1], x)
                + self.expected_inv_sigma_x()[..., -1:]
            )
            nat_params = ArrayDict(
                inv_sigma_mu=inv_sigma_mu, inv_sigma=self.expected_inv_sigma()
            )
            res = -0.5 * bdot(
                x.mT, bdot(self.expected_x_inv_sigma_x()[..., :-1, :-1], x)
            )
            res = (
                res
                - bdot(self.expected_x_inv_sigma_x()[..., -1:, :-1], x)
                - 0.5 * self.expected_x_inv_sigma_x()[..., -1:, -1:]
            )
        else:
            nat_params = ArrayDict(
                inv_sigma_mu=bdot(self.expected_inv_sigma_x(), x),
                inv_sigma=self.expected_inv_sigma(),
            )
            res = -0.5 * bdot(x.mT, bdot(self.expected_x_inv_sigma_x(), x))

        res += 0.5 * self.expected_logdet_inv_sigma()
        res = res.squeeze((-2, -1))
        return MultivariateNormal(nat_params, residual=res)

    def expected_inv_sigma_x(self, inv_sigma=None) -> Array:
        r"""
        Compute the expected value of the inverse covariance matrix times the input.

        .. math::
            E[\Sigma^{-1} A]
        """
        if inv_sigma is None:
            return bdot(self.expected_inv_sigma(), self.mu)
        else:
            return bdot(inv_sigma, self.mu)

    def expected_inv_sigma(self) -> Array:
        r"""
        Compute the expected value of the inverse covariance matrix.

        .. math::
            E[\Sigma^{-1}]
        """
        return self.u * self.n

    def expected_sigma(self) -> Array:
        r"""
        Compute the expected value of the covariance matrix.

        .. math::
            E[\Sigma]
        """
        return self.inv_u / (self.n - self.y_dim - 1)

    def inv_expected_inv_sigma(self) -> Array:
        r"""
        Compute the inverse of the expected value of the inverse covariance matrix.

        .. math::
            E[\Sigma^{-1}]^{-1}
        """
        return self.inv_u / self.n

    def expected_logdet_inv_sigma(self) -> Array:
        r"""
        Compute the expectation of the log determinant of the inverse covariance matrix.

        .. math::
            E[ log | \Sigma^{-1} |]
        """
        return (
            self.y_dim * jnp.log(2)
            - self.logdet_inv_u
            + mvdigamma(self.n / 2.0, self.y_dim)
        )

    def logdet_expected_inv_sigma(self) -> Array:
        r"""
        Compute the log determinant of the expected value of the inverse covariance matrix.

        .. math::
            log | E[\Sigma^{-1}] |
        """

        return jnp.log(self.n) - self.logdet_inv_u

    def expected_log_det_inv_sigma_minus_log_det_expected_inv_sigma(self) -> Array:
        return (
            self.y_dim * jnp.log(2)
            + mvdigamma(self.n / 2.0, self.y_dim)
            - jnp.log(self.n)
        )

    def residual_contrib(self):
        """
        The contribution to the residual that just comes from the undertainty over the linear transformation
        is given by expected_logdet_inv_sigma - logdet_expected_inv_sigma
        Note that this does not include contributions that interact with X and Y
        """
        return 0.5 * (
            self.y_dim * jnp.log(2)
            + mvdigamma(self.n / 2.0, self.y_dim)
            - jnp.log(self.n)
        ).squeeze((-2, -1))

    def expected_x_inv_sigma_x(self, inv_sigma_mu=None) -> Array:
        r"""
        Compute the expected value of the inverse covariance matrix times the input.

        .. math::
            E[A^T\Sigma^{-1} A]
        """
        if inv_sigma_mu is None:
            return self.y_dim * self.v + bdot(
                self.mu.mT, bdot(self.expected_inv_sigma(), self.mu)
            )
        else:
            return self.y_dim * self.v + bdot(self.mu.mT, inv_sigma_mu)

    def _update_cache(self):
        """Update parameters required for computing expectations"""
        self._v, self._logdet_inv_v = inv_and_logdet(self.inv_v)
        self._u, self._logdet_inv_u = inv_and_logdet(self.inv_u)

    def expected_log_likelihood(self, data: Tuple[Array]) -> Array:
        r"""
        Compute the expected log likelihood of the given data.

        .. math::
            E_{p(A,\Sigma)}[log p(y|x,A,\Sigma)]
        """
        x, y = data
        inv_sigma = self.expected_inv_sigma()
        EinvUX = self.expected_inv_sigma_x(inv_sigma=inv_sigma)
        EXTinvUX = self.expected_x_inv_sigma_x(inv_sigma_mu=EinvUX)

        logp = -0.5 * bdot(y.mT, bdot(inv_sigma, y)).squeeze((-1, -2))
        if self.use_bias:
            logp += bdot(
                y.mT, bdot(EinvUX[..., :, :-1], x) + EinvUX[..., :, -1:]
            ).squeeze((-1, -2))
            logp -= 0.5 * bdot(x.mT, bdot(EXTinvUX[..., :-1, :-1], x)).squeeze((-1, -2))
            logp -= bdot(EXTinvUX[..., -1:, :-1], x).squeeze((-1, -2))
            logp -= 0.5 * EXTinvUX[..., -1, -1]
        else:
            logp += bdot(y.mT, bdot(EinvUX, x)).squeeze((-1, -2))
            logp -= 0.5 * bdot(x.mT, bdot(EXTinvUX, x)).squeeze((-1, -2))

        logp += 0.5 * (self.expected_logdet_inv_sigma()).squeeze(
            (-1, -2)
        ) - 0.5 * self.y_dim * jnp.log(2 * jnp.pi)
        return logp

    def average_energy(self, inputs: Tuple[Distribution]) -> Array:
        r"""
        Compute the average energy term of the likelihood factor, aka
        .. math::
            E_{P(x)P(y)q(A,\Sigma)}[log p(y|x,A,\Sigma)]

        Plays the role of the expected_log_likelihood term when the inputs are distributions over the inputs and outputs, aka
        .. math::
            P(X), P(Y)

        Note that the only requires of the input messages is that they have the following expectations defined: pX.mean and pX.expected_xx()
        """

        pX, pY = inputs

        inv_sigma = self.expected_inv_sigma()
        exp_inv_sigma_x = self.expected_inv_sigma_x(inv_sigma=inv_sigma)
        exp_x_inv_sigma_x = self.expected_x_inv_sigma_x(inv_sigma_mu=exp_inv_sigma_x)

        logp = 0.5 * self.expected_logdet_inv_sigma().squeeze(
            (-1, -2)
        ) - 0.5 * self.y_dim * jnp.log(2 * jnp.pi)
        logp -= 0.5 * (pY.expected_xx() * inv_sigma).sum((-2, -1))
        if self.use_bias:
            inv_sigma_mu_x = (
                bdot(exp_inv_sigma_x[..., :, :-1], pX.mean)
                + exp_inv_sigma_x[..., :, -1:]
            )
            logp += jnp.sum(inv_sigma_mu_x * pY.mean.mT, axis=(-1, -2))

            logp -= 0.5 * (pX.expected_xx() * exp_x_inv_sigma_x[..., :-1, :-1]).sum(
                (-2, -1)
            )
            logp -= bdot(exp_x_inv_sigma_x[..., -1:, :-1], pX.mean).squeeze((-1, -2))
            logp -= 0.5 * exp_x_inv_sigma_x[..., -1, -1]
        else:
            inv_sigma_mu_x = bdot(exp_inv_sigma_x, pX.mean)
            logp += jnp.sum(inv_sigma_mu_x * pY.mean.mT, axis=(-1, -2))
            logp -= 0.5 * (pX.expected_xx() * exp_x_inv_sigma_x).sum((-1, -2))

        return logp

    def kl_divergence(self) -> Array:
        r"""
        Compute the KL divergence between the posterior and prior distributions.

        .. math::
            KL(p(A,\Sigma) || p(A,\Sigma)) = KL(p(A|\Sigma) || p(A|\Sigma)) + KL(p(\Sigma) || p(\Sigma))
        """
        kl = (
            self.y_dim / 2.0 * self.logdet_inv_v.squeeze((-2, -1))
            - self.y_dim / 2.0 * self.prior_logdet_inv_v.squeeze((-2, -1))
            - self.y_dim * self.x_dim / 2.0
        )

        traceV = (self.prior_inv_v * self.v).sum((-2, -1))
        kl += 0.5 * self.y_dim * traceV

        traceMuUMu = (
            self.prior_inv_v
            * bdot(
                (self.mu - self.prior_mu).mT,
                bdot(self.expected_inv_sigma(), self.mu - self.prior_mu),
            )
        ).sum((-2, -1))
        kl += 0.5 * traceMuUMu

        return kl + self._kl_divergence_wishart()

    def _kl_divergence_wishart(self) -> Array:
        r"""
        Compute the KL divergence between the posterior and prior wishart distributions.

        .. math::
            KL(p(\Sigma) || p(\Sigma))
        """
        kl = self.prior_n / 2.0 * (self.logdet_inv_u - self.prior_logdet_inv_u)  # (1,1)
        trace = (self.prior_inv_u * self.u).sum((-2, -1), keepdims=True)  # (1,1)
        kl += self.n / 2.0 * trace  # (1,1)
        kl -= self.n * self.y_dim / 2.0  # (1,1)
        kl = (
            kl
            + mvgammaln(self.prior_n / 2.0, self.y_dim)
            - mvgammaln(self.n / 2.0, self.y_dim)
            + (self.n - self.prior_n) / 2.0 * mvdigamma(self.n / 2.0, self.y_dim)
        )  # (1,1)
        return kl.squeeze((-1, -2))

    def forward_from_normal(
        self, pX: MultivariateNormal, pass_residual=False
    ) -> MultivariateNormal:
        r"""
        When the input distribution is a multivariate normal (or mixture of multivariate normals), then
        we can analytically compute the the forward message by marginalising over the input distribution.
        This is computationally more expensive, but gives better results.  The logic of this function is to
        take advantage of the fact that:
            exp\left( \left< \log p(y|x,A,\Sigma) \right>_{q(A,\Sigma)} \right)
            =
            p(y|x,<A>,<\Sigma^{-1}>^{-1}) \exp( \frac{1}{2} (< \log | \Sigma^{-1} |>  - \log | <\Sigma^{-1}> | - dim_y X^T V X ) )

        As a result the forward message is obtained by just adding some evidence for X with precision dim_y*V
        and then linearlly transforming it by <A> and then adding noise with covariance <\Sigma^{-1}>^{-1}
        """
        # The definition of residual used here is log q(y) + residual = int exp <log p(y|x,A,\Sigma)>_{q(A,\Sigma)} * q(x)dx

        # Stuff that doesnt go into joint goes into the residual (drop the measure function???)
        res = -pX.log_partition().squeeze((-1, -2))
        res += 0.5 * self.expected_logdet_inv_sigma().squeeze(
            (-2, -1)
        )  # - 0.5 * self.y_dim * jnp.log(2 * jnp.pi)
        if self.use_bias is False:
            # Joint Precision = [A, -B;-B^T, D]
            # Joint Precision@mu = [C_y; C_x]
            A = self.expected_inv_sigma()
            B = self.expected_inv_sigma_x()
            D = self.expected_x_inv_sigma_x() + pX.inv_sigma
            C_y = 0.0
            C_x = pX.inv_sigma_mu

            invD, logdetD = inv_and_logdet(D)
            # stuff that completes the square goes into the residual
            res += 0.5 * bdot(C_x.mT, bdot(invD, C_x)).squeeze(
                (-2, -1)
            ) - 0.5 * logdetD.squeeze((-2, -1))
        else:
            A = self.expected_inv_sigma()
            B = self.expected_inv_sigma_x()
            D = self.expected_x_inv_sigma_x()
            C_y = B[..., :, -1:]
            B = B[..., :, :-1]
            C_x = -D[..., :-1, -1:] + pX.inv_sigma_mu
            res += -0.5 * D[..., -1, -1]  # from bbT (doesnt go into the joint)
            D = D[..., :-1, :-1] + pX.inv_sigma

            invD, logdetD = inv_and_logdet(D)
            # stuff that completes the square goes into the residual
            res += 0.5 * bdot(C_x.mT, bdot(invD, C_x)).squeeze(
                (-2, -1)
            ) - 0.5 * logdetD.squeeze((-2, -1))

        inv_sigma_yy = A - bdot(B, bdot(invD, B.mT))
        inv_sigma_mu_y = C_y + bdot(B, bdot(invD, C_x))

        ndim_diff = inv_sigma_mu_y.ndim - inv_sigma_yy.ndim
        if ndim_diff > 0:
            inv_sigma_yy = jnp.expand_dims(inv_sigma_yy, tuple(range(ndim_diff)))

        if pass_residual:
            res += pX.residual
        pY = MultivariateNormal(
            ArrayDict(inv_sigma_mu=inv_sigma_mu_y, inv_sigma=inv_sigma_yy), residual=res
        )
        pY.residual += pY.log_partition().squeeze((-2, -1))
        return pY

    def backward_from_normal(
        self, pY: MultivariateNormal, pass_residual=False
    ) -> MultivariateNormal:
        """
        When the output distribution is a multivariate normal (or mixture of multivariate normals), then
        we can analytically compute the the backward message by marginalising over the output distribution.
        This is computationally more expensive, but gives better results.
        """
        # stuff that doesnt go into joint goes into the residual (drop measure???)
        res = -pY.log_partition().squeeze((-1, -2))
        res += 0.5 * self.expected_logdet_inv_sigma().squeeze(
            (-1, -2)
        )  # - 0.5 * self.y_dim * jnp.log(2 * jnp.pi)
        A = self.expected_inv_sigma() + pY.inv_sigma
        invA, logdetA = inv_and_logdet(A)

        if self.use_bias is False:
            # Joint Precision = [A, -B;-B^T, D]
            # Joint Precision@mu = [C_y; C_x]
            B = self.expected_inv_sigma_x()
            D = self.expected_x_inv_sigma_x()
            C_y = pY.inv_sigma_mu
            C_x = 0.0

            invA, logdetA = inv_and_logdet(A)
            res += 0.5 * bdot(C_y.mT, bdot(invA, C_y)).squeeze(
                (-2, -1)
            ) - 0.5 * logdetA.squeeze((-2, -1))
        else:
            B = self.expected_inv_sigma_x()
            D = self.expected_x_inv_sigma_x()
            C_y = pY.inv_sigma_mu + B[..., :, -1:]
            B = B[..., :, :-1]
            C_x = -D[..., :-1, -1:]
            res += -0.5 * D[..., -1, -1]  # from bbT (doesnt go into the joint)
            D = D[..., :-1, :-1]
            # stuff that completes the square goes into the residual
            res += 0.5 * bdot(C_y.mT, bdot(invA, C_y)).squeeze(
                (-2, -1)
            ) - 0.5 * logdetA.squeeze((-2, -1))

        inv_sigma_xx = D - bdot(B.mT, bdot(invA, B))
        inv_sigma_mu_x = C_x + bdot(B.mT, bdot(invA, C_y))

        ndim_diff = inv_sigma_mu_x.ndim - inv_sigma_xx.ndim
        if ndim_diff > 0:
            inv_sigma_xx = jnp.expand_dims(inv_sigma_xx, tuple(range(ndim_diff)))

        if pass_residual:
            res += pY.residual
        pX = MultivariateNormal(
            ArrayDict(inv_sigma_mu=inv_sigma_mu_x, inv_sigma=inv_sigma_xx), residual=res
        )
        return pX

    def backward_smooth_from_normal(
        self, pY: MultivariateNormal, pX: MultivariateNormal, pass_resiudal=False
    ) -> MultivariateNormal:
        return self.backward_from_normal(pY) * pX

    def variational_backward_smooth(
        self, pY: MultivariateNormal, pX: MultivariateNormal, pass_residual=False
    ) -> MultivariateNormal:
        return self.variational_backward(pY) * pX

    def forward(self, pX: Distribution, pass_residual=False) -> Distribution:
        return self.variational_forward(pX, pass_residual=pass_residual)

    def backward(self, pY: Distribution, pass_residual=False) -> Distribution:
        return self.variational_backward(pY, pass_residual=pass_residual)

    def variational_forward(
        self, pX: Distribution, pass_residual=False
    ) -> MultivariateNormal:

        # The definition of residual used here is log q(y) + residual = <log p(y|x,A,\Sigma)>_{q(A,\Sigma)q(x)}

        inv_sigma_y = self.expected_inv_sigma()
        res = 0.5 * self.expected_logdet_inv_sigma().squeeze((-1, -2))
        inv_sigma_x = bdot(inv_sigma_y, self.mu)
        inv_sigma_xx = self.y_dim * self.v + bdot(self.mu.mT, inv_sigma_x)
        if self.use_bias:
            inv_sigma_mu_y = (
                bdot(inv_sigma_x[..., :, :-1], pX.expected_x())
                + inv_sigma_x[..., :, -1:]
            )  # xy, yb terms

            res -= 0.5 * jnp.sum(
                inv_sigma_xx[..., :-1, :-1] * pX.expected_xx(), (-2, -1)
            )  # xx term
            res -= jnp.sum(
                inv_sigma_xx[..., :-1, -1:] * pX.expected_x(), (-2, -1)
            )  # xb term
            res -= 0.5 * inv_sigma_xx[..., -1, -1]  # bb term
        else:
            inv_sigma_mu_y = bdot(inv_sigma_x, pX.expected_x())  # xy term
            res -= 0.5 * jnp.sum(inv_sigma_xx * pX.expected_xx(), (-2, -1))  # xx term

        shape = jnp.broadcast_shapes(inv_sigma_y.shape[:-2], inv_sigma_mu_y.shape[:-2])
        inv_sigma_y = jnp.broadcast_to(inv_sigma_y, shape + inv_sigma_y.shape[-2:])
        inv_sigma_mu_y = jnp.broadcast_to(
            inv_sigma_mu_y, shape + inv_sigma_mu_y.shape[-2:]
        )

        if pass_residual:
            res += pX.residual
        pY = MultivariateNormal(
            ArrayDict(inv_sigma_mu=inv_sigma_mu_y, inv_sigma=inv_sigma_y), residual=res
        )
        pY.residual += pY.log_partition().squeeze((-1, -2))
        return pY

    def variational_backward(
        self, pY: Distribution, pass_residual=False
    ) -> Distribution:
        # uses definition of residual where nat_params*T(x) + residual + log_measure(y) = <log p(y|x,A,\Sigma)>_{q(A,\Sigma)q(y)}
        # this is different from the variational_backward residual definition which includes the log partition function of the
        # backward likelihood.  Jeff thinks this should not be done because this is a likelhood function not a posterior
        #
        res = 0.5 * self.expected_logdet_inv_sigma().squeeze((-2, -1))
        inv_sigma = self.expected_inv_sigma()
        res -= 0.5 * jnp.sum(inv_sigma * pY.expected_xx(), (-2, -1))  # yy term
        if self.use_bias is False:
            inv_sigma_mu = self.expected_inv_sigma_x(inv_sigma=inv_sigma)
            inv_sigma_mu_x = bdot(inv_sigma_mu.mT, pY.expected_x())  # xy term
            inv_sigma_xx = self.expected_x_inv_sigma_x(
                inv_sigma_mu=inv_sigma_mu
            )  # xx term
        else:
            Einv_sigma_x = self.expected_inv_sigma_x(inv_sigma=inv_sigma)
            Ex_inv_sigma_x = self.expected_x_inv_sigma_x(inv_sigma_mu=Einv_sigma_x)
            inv_sigma_mu_x = (
                bdot(Einv_sigma_x[..., :, :-1].mT, pY.expected_x())
                - Ex_inv_sigma_x[..., :-1, -1:]
            )  # yx and bx term
            res += (
                bdot(Einv_sigma_x[..., :, -1:].mT, pY.expected_x()).squeeze((-2, -1))
                - 0.5 * Ex_inv_sigma_x[..., -1, -1]
            )  # yb and bb term
            inv_sigma_xx = Ex_inv_sigma_x[..., :-1, :-1]  # xx term

        shape = jnp.broadcast_shapes(
            inv_sigma_xx.shape[:-2], pY.expected_x().shape[:-2]
        )
        inv_sigma_xx = jnp.broadcast_to(inv_sigma_xx, shape + inv_sigma_xx.shape[-2:])
        pX = MultivariateNormal(
            ArrayDict(inv_sigma_mu=inv_sigma_mu_x, inv_sigma=inv_sigma_xx), residual=res
        )
        if pass_residual:
            pX.residual += pY.residual
        return pX

    def variational_backward_new(
        self, pY: Distribution, pass_residual=False
    ) -> Distribution:
        """
        More generally, i.e. when the output distribution is not multivariate normal (or mixture of multivariate
        normals), then an analytic computation of the backward message is not possible. Instead, we use a variational
        approximation for the backward message.  This approximation, leads to an larger residual term, but is
        computationally cheaper, even when the input distribution is normally distributed.
        """
        res = -0.5 * (self.y_dim - (self.x_dim - self.use_bias)) * jnp.log(2 * jnp.pi)
        res += 0.5 * self.expected_logdet_inv_sigma().squeeze((-2, -1))
        inv_sigma_y = self.expected_inv_sigma()
        res -= 0.5 * jnp.sum(inv_sigma_y * pY.expected_xx(), (-2, -1))  # yy term
        inv_sigma_x = bdot(inv_sigma_y, self.mu)
        inv_sigma_xx = self.y_dim * self.v + bdot(self.mu.mT, inv_sigma_x)  # xx term

        shape = jnp.broadcast_shapes(
            inv_sigma_xx.shape[:-2], pY.expected_x().shape[:-2]
        )
        inv_sigma_xx = jnp.broadcast_to(inv_sigma_xx, shape + inv_sigma_xx.shape[-2:])

        if self.use_bias:
            _inv_sigma_xx = inv_sigma_xx[..., :-1, :-1]
            chol = jsp.linalg.cho_factor(_inv_sigma_xx, lower=True)
            logdet_inv_sigma_xx = jnp.log(
                jnp.diagonal(chol[0], axis1=-1, axis2=-2)
            ).sum(-1)
            res -= logdet_inv_sigma_xx

            inv_sigma_mu_x = (
                bdot(inv_sigma_x[..., :, :-1].mT, pY.expected_x())
                - inv_sigma_xx[..., :-1, -1:]
            )  # xy,xb terms

            mu_x = jsp.linalg.cho_solve(chol, inv_sigma_mu_x)
            res += 0.5 * jnp.sum(mu_x * inv_sigma_mu_x, (-2, -1))  # xx term
            res += 0.5 * jnp.sum(
                inv_sigma_x[..., :, -1:] * pY.expected_x(), (-2, -1)
            )  # yb term
            res -= 0.5 * inv_sigma_xx[..., -1, -1]  # bb term
            inv_sigma_xx = _inv_sigma_xx
        else:
            chol = jsp.linalg.cho_factor(inv_sigma_xx, lower=True)
            logdet_inv_sigma_xx = jnp.log(
                jnp.diagonal(chol[0], axis1=-1, axis2=-2)
            ).sum(-1)
            res -= logdet_inv_sigma_xx
            inv_sigma_mu_x = bdot(inv_sigma_x.mT, pY.expected_x())  # xy term
            mu_x = jsp.linalg.cho_solve(chol, inv_sigma_mu_x)
            res += 0.5 * jnp.sum(mu_x * inv_sigma_mu_x, (-2, -1))

        pX = MultivariateNormal(
            ArrayDict(inv_sigma_mu=inv_sigma_mu_x, inv_sigma=inv_sigma_xx), residual=res
        )
        if pass_residual:
            pX.residual += pY.residual
        return pX

    def joint(self, pX: Distribution, pY: Distribution) -> Distribution:
        """
        computes the joint distribution of the input and output given the model parameters
        This is used to compute the equivalent of xi in the HMM case and is needed for exact
        inference in the case of linear dynamical systems.
        """
        raise NotImplementedError

    def elbo(self, data: Tuple[Array], weights: Optional[Array] = None) -> Array:
        """
        Compute the evidence lower bound of the model given the data.
        """
        X, Y = data
        sample_dims = self.get_sample_dims(X)
        if weights is None:
            ELL = self.expected_log_likelihood((X, Y)).sum(sample_dims)
        else:
            ELL = (self.expected_log_likelihood(data) * weights).sum(sample_dims)
        return ELL - self.kl_divergence()

    def elbo_contrib(
        self, pXY: Tuple[Distribution], weights: Optional[Array] = None
    ) -> Array:
        """
        Compute the evidence lower bound of the model given the data.
        """
        pX, pY = pXY
        sample_dims = self.get_sample_dims(pX.mean)
        if weights is None:
            ELL = self.average_energy(pXY).sum(sample_dims)
        else:
            ELL = (self.average_energy(pXY) * weights).sum(sample_dims)
        return ELL - self.kl_divergence()


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    key = jr.PRNGKey(11)
    ns = 400
    batch_shape = ()
    y_dim = 4
    x_dim = 4
    scale = 0.1
    use_bias = True
    fixed_precision = False

    num_samples = 1000
    #    W = jr.normal(key,batch_shape + (y_dim,x_dim))/jnp.sqrt(x_dim)
    #    X = jr.normal(key,(ns,) + batch_shape + (x_dim,1))
    #    B = jr.normal(key,batch_shape + (y_dim,1))
    W = jr.normal(key, (y_dim, x_dim)) / jnp.sqrt(x_dim)
    X = jr.normal(key, (ns,) + (x_dim, 1))
    B = jr.normal(key, (y_dim, 1)) * use_bias
    W = W * X.std()
    X = X / X.std()
    Y = W @ X + B * use_bias

    for i in range(len(batch_shape)):
        X = jnp.expand_dims(X, -3)
        Y = jnp.expand_dims(Y, -3)

    prior_mu = jnp.full(
        batch_shape + (y_dim, x_dim + use_bias), 0.0
    )  # shape should be (batch_shape,) + (y_dim, x_dim)
    prior_n = jnp.full(
        batch_shape + (1, 1), 2.0 + y_dim
    )  # shape should be (batch_shape,) + (1, 1)
    prior_inv_v = jnp.broadcast_to(
        jnp.eye(x_dim + use_bias), batch_shape + (x_dim + use_bias, x_dim + use_bias)
    )  # shape should be (batch_shape,) + (x_dim, x_dim). Same goes for the variable `v`
    prior_inv_u = scale**2 * jnp.broadcast_to(
        jnp.eye(y_dim), batch_shape + (y_dim, y_dim)
    )  # shape should be (batch_shape,) + (y_dim, y_dim). Same goes for the variable `inv_sigma` and `inv_expected_inv_sigma()`
    mu = prior_mu + jr.normal(key, batch_shape + (y_dim, x_dim + use_bias)) / jnp.sqrt(
        x_dim
    )

    params = ArrayDict(mu=mu, inv_v=prior_inv_v, inv_u=prior_inv_u, n=prior_n)
    prior_params = ArrayDict(
        mu=prior_mu, inv_v=prior_inv_v, inv_u=prior_inv_u, n=prior_n
    )

    model = Linear(
        params, prior_params, use_bias=use_bias, fixed_precision=fixed_precision
    )

    weights = jr.uniform(key, (ns,) + batch_shape)
    model.update_from_data((X, Y), weights=weights)
    pX = MultivariateNormal(
        ArrayDict(inv_sigma_mu=100 * X, inv_sigma=100 * jnp.eye(X.shape[-2]))
    )
    pY = MultivariateNormal(
        ArrayDict(inv_sigma_mu=100 * Y, inv_sigma=100 * jnp.eye(Y.shape[-2]))
    )

    pYhat = model.predict(X)
    plt.scatter(Y.squeeze(), pYhat.mean.squeeze())
    plt.title("predict")
    plt.show()

    pXhat = model.variational_backward(pYhat)
    plt.scatter(X.squeeze(), pXhat.mean.squeeze())
    plt.title("variational backward")
    plt.show()

    pXhat = model.backward_from_normal(pYhat)
    plt.scatter(X.squeeze(), pXhat.mean.squeeze())
    plt.title("backward_from_normal")
    plt.show()

    pYhat = model.forward_from_normal(pXhat)
    plt.scatter(Y.squeeze(), pYhat.mean.squeeze())
    plt.title("forward_from_normal")
    plt.show()

    pYhat = model.variational_forward(pXhat)
    plt.scatter(Y.squeeze(), pYhat.mean.squeeze())
    plt.title("variational_forward")
    plt.show()

    model_2 = Linear(
        params, prior_params, use_bias=use_bias, fixed_precision=fixed_precision
    )

    model_2.update_from_probabilities((pX, pY), weights=weights)

    pYhat = model_2.forward_from_normal(pX)
    plt.scatter(Y.squeeze(), pYhat.mean.squeeze())
    plt.title("forward_from_normal")
    plt.show()

    pXhat = model_2.backward_from_normal(pY)
    plt.scatter(X.squeeze(), pXhat.mean.squeeze())
    plt.title("backward_from_normal")
    plt.show()

    elbopxpy = model_2.elbo_contrib((pXhat, pYhat))
    elbo = model.elbo((X, Y))

    ellpxpy = model_2.average_energy((pX, pY))
    ell = model_2.expected_log_likelihood((X, Y))

    plt.scatter(ell, ellpxpy)
    plt.show()

    # px = model.variational_backward(pY)
    # px2 = model.variational_backward_new(pY)
    # plt.scatter(px.residual + px.log_partition().squeeze((-2,-1)), px2.residual)
    # ((px.residual.mean() + px.log_partition().squeeze((-2,-1)) - px2.residual).mean())*2/jnp.log(2*jnp.pi)

    pYhat = model.variational_forward(pX)
    pXhat = model.variational_backward(pY)
    elbo = model.average_energy((pX, pY))

    #    log q(y) + residual = <log p(y|x,A,\Sigma)>_{q(A,\Sigma)q(x)}
    #    <log q(y)>_p(y) + residual = average energy

    def E_eta_dot_T(p, q):
        return -0.5 * jnp.sum(p.expected_xx() * q.inv_sigma, (-2, -1)) + jnp.sum(
            p.expected_x() * q.inv_sigma_mu, (-2, -1)
        )  # + q.log_partition().squeeze((-1, -2))

    Elog_q = (
        E_eta_dot_T(pY, pYhat)
        - pYhat.log_partition().squeeze((-1, -2))
        - 0.5 * y_dim * jnp.log(2 * jnp.pi)
    )

    plt.scatter(elbo, Elog_q + pYhat.residual)
    plt.show()

    plt.scatter(
        elbo + 0.5 * y_dim * jnp.log(2 * jnp.pi),
        E_eta_dot_T(pX, pXhat) + pXhat.residual,
    )
    plt.show()

    plt.scatter(
        elbo + 0.5 * y_dim * jnp.log(2 * jnp.pi),
        E_eta_dot_T(pY, pYhat)
        + pYhat.residual
        - pYhat.log_partition().squeeze((-1, -2)),
    )
    plt.show()

    pYhat2 = model.forward_from_normal(pX)
    pXhat2 = model.backward_from_normal(pY)

    plt.scatter(pYhat.residual, pYhat2.residual)
    plt.title("forward residuals")
    plt.show()
    plt.scatter(
        pXhat.residual + pXhat.log_partition().squeeze((-2, -1)),
        pXhat2.residual + pXhat2.log_partition().squeeze((-2, -1)),
    )
    plt.title("backward residuals + log partition")
    plt.show()
