# Copyright 2024 VERSES AI, Inc.
#
# Licensed under the VERSES Academic Research License (the “License”);
# you may not use this file except in compliance with the license.
#
# You may obtain a copy of the License at
#
#     https://github.com/VersesTech/cavi-cmn/blob/main/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Union, Optional, Dict, Tuple
from functools import partial
from jax import lax, jit
import jax
import jax.tree_util as jtu
import jax.numpy as jnp
from jax.numpy import expand_dims as expand
from jax.nn import softmax
from jax.scipy.special import logsumexp
from jax.scipy.linalg import solve
from jaxtyping import Array

from cavi_cmn.utils import symmetrise
from cavi_cmn.utils import make_posdef

from multimethod import multimethod

from cavi_cmn import ArrayDict, Distribution, Delta
from cavi_cmn.models import Model
from cavi_cmn.transforms import Linear, LinearMatrixNormalGamma, MultinomialRegression
from cavi_cmn.utils import map_and_multiply
from cavi_cmn.exponential import ExponentialFamily as ExpFam
from cavi_cmn.exponential import MultivariateNormal as ExpMVN
from cavi_cmn.exponential import Multinomial as Cat
from cavi_cmn.exponential import MixtureMessage


class ConditionalMixture(Model):
    pytree_data_fields = ("likelihood", "pi")
    pytree_aux_fields = ("pi_opts", "likelihood_opts", "average_type")

    def __init__(
        self,
        likelihood=None,
        pi=None,
        likelihood_params=None,
        pi_params=None,
        pi_opts: Optional[Dict] = None,
        likelihood_opts: Optional[Dict] = None,
        average_type: str = "statistics",
        likelihood_prior_type: str = "mnw",
    ):

        self.pi_opts = (
            pi_opts if pi_opts is not None else {"iters": 1, "lr": 1.0, "beta": 0.0}
        )
        self.likelihood_opts = (
            likelihood_opts if likelihood_opts is not None else {"lr": 1.0, "beta": 0.0}
        )

        if isinstance(likelihood, Distribution) and likelihood_params is not None:
            print(
                "Warning: provided `likelihood` is a Distribution, so `likelihood_params` will be ignored"
            )
        elif not isinstance(likelihood, Distribution) and likelihood_params is not None:
            if likelihood_prior_type == "mnw":
                likelihood = Linear(
                    *likelihood_params["args"], **likelihood_params["kwargs"]
                )
            elif likelihood_prior_type == "mng":
                likelihood = LinearMatrixNormalGamma(
                    *likelihood_params["args"], **likelihood_params["kwargs"]
                )
            else:
                raise NotImplementedError

        if isinstance(pi, Distribution) and pi_params is not None:
            print(
                "Warning: provided `pi` is a Distribution, so `pi_params` will be ignored"
            )
        elif not isinstance(pi, Distribution) and pi_params is not None:
            pi = MultinomialRegression(
                *pi_params["args"], **pi_params["kwargs"], optim_args=self.pi_opts
            )

        self.average_type = average_type
        batch_shape = pi.batch_shape
        event_shape = likelihood.event_shape
        super().__init__(likelihood.default_event_dim, batch_shape, event_shape)

        self.likelihood = likelihood
        self.pi = pi
        _ = self.pi.beta.expectations

    def update_from_data(self, inputs: Tuple[Array], iters: int = 1):
        self.likelihood, self.pi, elbo = self._update_from_data(
            inputs, self.likelihood, self.pi, iters=iters
        )
        return elbo

    # @partial(jit, static_argnames=["iters"])
    def _update_from_data(
        self,
        inputs: Tuple[Array],
        likelihood: Distribution,
        pi: Distribution,
        iters: int,
    ):

        x, y = inputs
        sample_dims = tuple(
            range(len(x.shape) - self.batch_dim - self.likelihood.event_dim)
        )
        mix_dims = tuple(range(-self.pi.event_dim, 0))
        like_mix_dims = tuple(
            range(
                -self.pi.event_dim - self.likelihood.event_dim,
                -self.likelihood.event_dim,
            )
        )
        x = jnp.expand_dims(x, like_mix_dims)
        y = jnp.expand_dims(y, like_mix_dims)

        def step_fn(carry, _):
            likelihood, pi = carry

            # elbo computation
            log_probs = likelihood.expected_log_likelihood((x, y))
            log_probs = log_probs + pi.log_predict(x.squeeze(like_mix_dims))
            assignments = softmax(log_probs, mix_dims)

            log_z = logsumexp(log_probs, mix_dims)
            elbo = (
                jnp.sum(log_z, sample_dims)
                - likelihood.kl_divergence().sum(mix_dims)
                - pi.kl_divergence()
            )

            # M step
            pi.update((x.squeeze(like_mix_dims), assignments))
            likelihood.update_from_data((x, y), assignments, **self.likelihood_opts)
            return (likelihood, pi), elbo

        init_distributions = (likelihood, pi)
        (likelihood, pi), elbos = lax.scan(
            step_fn, init_distributions, jnp.arange(iters)
        )

        return likelihood, pi, elbos

    def get_assignments_from_data(self, inputs: Tuple[Array]) -> Array:
        x, y = inputs
        like_mix_dims = tuple(
            range(
                -self.pi.event_dim - self.likelihood.event_dim,
                -self.likelihood.event_dim,
            )
        )
        x = jnp.expand_dims(x, like_mix_dims)
        y = jnp.expand_dims(y, like_mix_dims)
        return softmax(
            self.likelihood.expected_log_likelihood((x, y))
            + self.pi.log_predict(x.squeeze(like_mix_dims))
        )

    def get_assignments_from_probabilities(self, inputs: Tuple[Distribution]) -> Array:
        pX, pY = inputs
        pX_with_batch = pX.expand_batch_shape(-1)
        pY_with_batch = pY.expand_batch_shape(-1)
        return softmax(
            self.likelihood.average_energy((pX_with_batch, pY_with_batch))
            + self.pi.log_forward(pX)
        )

    def update_from_probabilities(self, inputs: Tuple[Distribution], iters: int = 1):
        self.likelihood, self.pi, elbo = self._update_from_probabilities(
            inputs, self.likelihood, self.pi, iters=iters
        )
        return elbo

    # @partial(jit, static_argnames=["iters"])
    def _update_from_probabilities(
        self,
        inputs: Tuple[Distribution],
        likelihood: Distribution,
        pi: Distribution,
        iters: int,
    ):

        pX, pY = inputs

        if isinstance(pX, MixtureMessage):
            pX = pX.marginalize_statistics(keepdims=False)

        sample_dims = tuple(
            range(len(pX.shape) - self.batch_dim - self.likelihood.event_dim)
        )
        mix_dims = tuple(range(-self.pi.event_dim, 0))
        pX_with_batch = pX.expand_batch_shape(mix_dims)

        if isinstance(pY, MixtureMessage):
            self.pi.update((pX, pY.assignments))
            self.likelihood.update_from_probabilities(
                (pX_with_batch, pY.likelihood),
                weights=pY.assignments.mean,
                **self.likelihood_opts,
            )
            return self.likelihood, self.pi, 0.0

        pY_with_batch = pY.expand_batch_shape(mix_dims)

        def step_fn(carry, _):
            likelihood, pi = carry

            # elbo computation
            log_probs = likelihood.average_energy((pX_with_batch, pY_with_batch))
            log_probs = log_probs + pi.log_forward(pX)
            assignments = softmax(log_probs, mix_dims)

            log_z = logsumexp(log_probs, mix_dims)
            elbo = (
                jnp.sum(log_z, sample_dims)
                - likelihood.kl_divergence().sum(mix_dims)
                - pi.kl_divergence()
            )

            # M step
            pi.update((pX, assignments))
            likelihood.update_from_probabilities(
                (pX_with_batch, pY_with_batch), assignments, **self.likelihood_opts
            )
            return (likelihood, pi), elbo

        init_distributions = (likelihood, pi)
        (likelihood, pi), elbo = lax.scan(
            step_fn, init_distributions, jnp.arange(iters)
        )
        return likelihood, pi, elbo

    def predict(self, x: jnp.array, average_type: str = "statistics"):
        logits = self.pi.log_predict(x)
        p_assignments = Cat(ArrayDict(logits=logits))
        pY = self.likelihood.predict(jnp.expand_dims(x, -3))

        return MixtureMessage(
            likelihood=pY, assignments=p_assignments, average_type=average_type
        )

    def forward_from_normal(self, pX):
        """
        Forward message for Conditional mixture of linear transforms
        :param pX: Distribution
        :return: MixtureMessage (Distribution)
        """
        multinomial_message = self.pi.forward(pX)

        pY = self.likelihood.forward_from_normal(pX.expand_batch_shape(-1))
        return MixtureMessage(likelihood=pY, assignments=multinomial_message)

    def forward_avg_natparams(self, pX: MixtureMessage):
        """
        Forward message for Conditional mixture of linear transforms
        :param pX: MixtureMessage (Distribution)
        :return: MixtureMessage (Distribution)

        This is identical to `forward(self, pX: MixtureMessage)` except that it first marginalizes the **natural parameters** of the input Distribution (a mixture distribution) and then calls the forward method on the marginalized distribution `pX_marg`
        """

        pX_marg = pX.marginalize_nat_params(keepdims=False)

        return self.forward(pX_marg, average_type="nat_params")

    def forward_avg_statistics(self, pX: MixtureMessage):
        """
        Forward message for Conditional mixture of linear transforms
        :param pX: MixtureMessage (Distribution)
        :return: MixtureMessage (Distribution)

        This is identical to `forward(self, pX: MixtureMessage)` except that it first marginalizes the **expected statistics** of the input Distribution (a mixture distribution) and then calls the forward method on the marginalized distribution `pX_marg`
        """

        pX_marg = pX.marginalize_statistics(keepdims=False)

        return self.forward(pX_marg, average_type="statistics")

    @multimethod
    def forward(self, pX: MixtureMessage):
        """
        Forward message for Conditional mixture of linear transforms
        :param pX: MixtureMessage (Distribution)
        :return: MixtureMessage (Distribution)

        This first marginalizes statistics of the input Distribution (a mixture distribution) and then calls the forward method on the marginalized distribution `pX_marg`
        """

        pX_marg = pX.marginalize_statistics(
            keepdims=False
        )  # could also average out pX.likelihood.sigma instead of expected_xx() -- @NOTE: try both

        return self.forward(pX_marg)

    @multimethod
    def forward_old(self, pX: Distribution):
        """
        Forward message for Conditional mixture of linear transforms
        :param pX: ExponentialFamily (Distribution)
        :return: MixtureMessage
        """

        pY = self.likelihood.variational_forward(pX.expand_batch_shape(-1))

        logits = (
            self.pi.log_forward(pX) + pY.residual + pY.log_partition().squeeze((-2, -1))
        )  # this assumes variational posterior of the form q(x,z) = q(x|z)q(z)

        pY.residual = logsumexp(logits, axis=-1, keepdims=True)
        return MixtureMessage(
            likelihood=pY,
            assignments=Cat(ArrayDict(logits=logits)),
            average_type=self.average_type,
        )

    @multimethod
    def forward(self, pX: Distribution):
        """
        Forward message for Conditional mixture of linear transforms
        :param pX: ExponentialFamily (Distribution)
        :return: MixtureMessage
        """

        pY = self.likelihood.variational_forward(pX.expand_batch_shape(-1))
        logits = self.pi.log_forward(pX)

        return MixtureMessage(
            likelihood=pY,
            assignments=Cat(ArrayDict(logits=logits)),
            average_type=self.average_type,
        )

    def backward_from_normal(self, pY):
        return self.backward(pY, collapsed=False)

    @multimethod
    def backward(self, pY: Distribution):
        sample_ndims = len(pY.shape) - self.likelihood.event_dim - self.pi.batch_dim

        ll_z = self.likelihood.variational_backward(pY.expand_batch_shape(-1))
        # shape (sample x batch x mixdim x xdim x 1)

        averaged_nat_params = ArrayDict(
            inv_sigma_mu=ll_z.inv_sigma_mu.mean(axis=-3, keepdims=False),
            inv_sigma=ll_z.inv_sigma.mean(axis=-3, keepdims=False),
        )  # summing out z_{l+1} dimension

        ll = ExpMVN(averaged_nat_params)

        dim = self.pi.y_dim
        p_z_logits = jnp.expand_dims(
            jnp.zeros(dim), tuple(range(sample_ndims))
        )  # create a uniform categorical variable

        p_z = Cat(ArrayDict(logits=p_z_logits))
        ll_pi_x = self.pi.backward(p_z)[0]

        return ll * ll_pi_x

    @multimethod
    def backward(self, pY: MixtureMessage):
        """
        Backward message for Conditional mixture of linear transforms
        :param pY: MixtureMessage (Distribution)
        :return: MixtureMessage (Distribution)
        """
        p_x_z, p_z = pY.likelihood, pY.assignments  # q(x_l, z_l)
        ll_z = self.likelihood.variational_backward(
            p_x_z
        )  # the backwards message $\propto q(x_l|z_l)$

        ll_pi_x = self.pi.joint(ll_z.moveaxis(-1, 0), categorical_axis=True)[0]

        # marginalize out the backward messsages for each value of z_{l}, using their posterior assignment probabilities under q(z_l)
        return MixtureMessage(
            likelihood=ll_pi_x, assignments=p_z
        ).marginalize_nat_params(keepdims=False)

    def merge_messages(
        self,
        forward_msg: MixtureMessage,
        backward_msg: Distribution,
        return_elbo_contribs: bool = False,
    ):
        q_x_z = forward_msg.likelihood * backward_msg.expand_batch_shape(-1)

        # elbo contrib computation
        eta = backward_msg.expand_batch_shape(-1).nat_params
        dim, mapping = q_x_z.default_event_dim, q_x_z._get_params_to_stats_mapping()
        Tx = jtu.tree_map(lambda x: x.mT, q_x_z.expected_statistics())
        elbo_contrib_bwd = map_and_multiply(eta, Tx, dim, mapping).squeeze((-1, -2))

        logits = (
            forward_msg.assignments.logits
            + forward_msg.likelihood.residual
            - forward_msg.likelihood.log_partition().squeeze((-1, -2))
            + q_x_z.log_partition().squeeze((-1, -2))
        )
        q_z_l = Cat(ArrayDict(logits=logits))

        elbo_contribs = q_z_l.log_normalizer.squeeze(-1) - jnp.sum(
            q_z_l.mean * elbo_contrib_bwd, -1
        )

        if return_elbo_contribs:
            return (
                MixtureMessage(
                    likelihood=q_x_z, assignments=q_z_l, average_type=self.average_type
                ),
                elbo_contribs,
            )
        else:
            return MixtureMessage(
                likelihood=q_x_z, assignments=q_z_l, average_type=self.average_type
            )

    @multimethod
    def backward_old(self, pY: MixtureMessage, collapsed=False, version_1=True):
        """
        Backward message for Conditional mixture of linear transforms
        :param pY: MixtureMessage (Distribution)
        :param collapsed: bool
        :param version_1: bool
        :return: MixtureMessage (Distribution)
        """
        pY_marg = pY.marginalize_statistics(keepdims=False)
        return self.backward(
            pY_marg, collapsed=collapsed, version_1=version_1
        )  # if the input is a tuple, we just ignore the second element and call the other backward method

    @multimethod
    def backward_old(self, pY: Distribution, collapsed=False, version_1=True):
        sample_ndims = len(pY.shape) - self.likelihood.event_dim - self.pi.batch_dim
        if collapsed:
            pX = self.likelihood.backward_from_normal(pY.expand_batch_shape(-1))
        else:
            pX = self.likelihood.variational_backward(pY.expand_batch_shape(-1))
        # shape (sample x batch x mixdim x xdim x 1)

        dim = self.pi.y_dim + 1
        if version_1:  #  VERSION 1 FAST!
            logits = jnp.expand_dims(
                jnp.log(jnp.eye(dim)),
                tuple(range(1, 1 + sample_ndims + self.batch_dim)),
            )
            # num_classes, (1,) * n_sample_dims + (1,) * batch_dims
            pZ = Cat(ArrayDict(logits=logits))
            pX2 = self.pi.backward(pZ, iters=2)  # does not push residual
            for i in range(len(self.batch_shape) + sample_ndims):
                pX2 = pX2.swap_axes(i, i + 1)
            pX = pX2 * pX
            log_p = pX.residual + pX.log_partition().squeeze((-2, -1))
        else:  # # VERSION 2 SLOW!
            logits = jnp.expand_dims(
                jnp.log(jnp.eye(dim)),
                tuple(range(1, 1 + sample_ndims + self.batch_dim)),
            )
            pZ = Cat(
                ArrayDict(logits=logits)
            )  # (num_classes, 1 * (sample_dims), 1 * (batch_dims), num_classes)
            residual = pX.residual  # pX will have shape (5, 4, 3, 2, 1, x_dim, 1)

            # this is a way to get around the lack of moveaxis on Distribution objects
            for i in range(len(self.batch_shape) + sample_ndims):
                pX = pX.swap_axes(-3 - i, -3 - i - 1)
                # goes from --> (5, 4, 3, 2, 1, x_dim, 1)
                # to --> (1, 5, 4, 3, 2, x_dim, 1)
            pX = self.pi.backward(pZ, pX, iters=2)  # (num_classes, 1, x_dim, 1)
            for i in range(len(self.batch_shape) + sample_ndims):
                pX = pX.swap_axes(i, i + 1)  # (..., num_classes, x_dim, 1)
            log_p = residual + pX.residual + pX.log_partition().squeeze((-2, -1))

        # logZ = logsumexp(log_p, axis=-1, keepdims=True)
        p = Cat(ArrayDict(logits=log_p))
        # pX.residual = logZ # added this logZ into the residual to be consistent with original way we were doing it

        return MixtureMessage(
            likelihood=pX, assignments=p, average_type=self.average_type
        )

    def expected_log_likelihood_given_pX_pY(self, pX, pY):
        pAX = pX.expand_batch_shape(-1)  # make mixture dimension compatible
        pAY = pY.expand_batch_shape(-1)
        log_p = self.likelihood.average_energy((pAX, pAY)) + self.pi.log_forward(pX)
        return logsumexp(log_p, axis=-1)

    def expected_log_likelihood(self, X, Y):
        log_p = self.likelihood.expected_log_likelihood(
            (expand(X, -3), expand(Y, -3))
        ) + self.pi.log_predict(X)
        return logsumexp(log_p, axis=-1)

    def expand_to_categorical_dims(self, x: Array, y: Array) -> Tuple[Array, Array]:
        x = jnp.reshape(
            x,
            x.shape[: -self.likelihood.event_dim]
            + (1,) * self.pi.event_dim
            + x.shape[-self.likelihood.event_dim :],
        )
        y = jnp.reshape(
            y,
            y.shape[: -self.likelihood.event_dim]
            + (1,) * self.pi.event_dim
            + y.shape[-self.likelihood.event_dim :],
        )
        return x, y

    def kl_divergence(self):
        return (
            self.likelihood.kl_divergence().sum(self.pi.get_event_dims())
            + self.pi.kl_divergence()
        )

    def elbo(self, X, Y):
        sample_dims = list(range(X.ndim - self.event_dim))
        return (
            self.expected_log_likelihood(X, Y).sum(sample_dims) - self.kl_divergence()
        )

    def elbo_contrib(self, pX, pY):
        sample_dims = list(range(len(pX.shape) - 2))
        return (
            self.expected_log_likelihood_given_pX_pY(pX, pY).sum(sample_dims)
            - self.kl_divergence()
        )
