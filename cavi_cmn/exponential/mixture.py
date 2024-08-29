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

from typing import Optional, Union, Tuple
from jaxtyping import Array
from multimethod import multimethod

from jax import numpy as jnp
import jax.tree_util as jtu

from cavi_cmn import ArrayDict
from cavi_cmn.distribution import Distribution, Delta
from cavi_cmn.exponential import ExponentialFamily, Multinomial
from cavi_cmn.utils import tree_marginalize, map_and_multiply


class MixtureMessage(Distribution):
    """
    This represents a Mixture of ExponentialFamily distributions, where
    the mixing distribution instance is a Categorical distribution.
    """

    pytree_data_fields = ("likelihood", "assignments")
    pytree_aux_fields = ("like_mix_dims", "average_type")

    def __init__(
        self,
        likelihood: ExponentialFamily,
        assignments: Optional[Multinomial] = None,
        average_type: str = "nat_params",
    ):

        super().__init__(
            likelihood.event_dim,
            likelihood.batch_shape,
            event_shape=likelihood.event_shape,
        )

        self.likelihood = likelihood
        if assignments is None:
            assignments = Multinomial(
                nat_params=ArrayDict(logits=jnp.zeros(likelihood.batch_shape + (1,)))
            )  # create a trivial mixture with one component
        self.assignments = assignments
        self.like_mix_dims = tuple(
            range(-self.assignments.event_dim - self.event_dim, -self.event_dim)
        )
        self.average_type = average_type

    def marginalize(self, keepdims=False) -> ExponentialFamily:
        """
        This returns the marginalized distribution of the likelihood distribution `self.likelihood`, using the posterior assignment probabilities given by `self.assignments`.
        """
        if self.average_type == "nat_params":
            return self.marginalize_nat_params(keepdims=keepdims)
        elif self.average_type == "statistics":
            return self.marginalize_statistics(keepdims=keepdims)
        else:
            raise ValueError(f"Invalid average type {self.average_type}")

    def marginalize_statistics(self, keepdims=False):
        assignment_probs = jnp.expand_dims(
            self.assignments.mean, axis=tuple(range(-self.event_dim, 0))
        )
        expected_stats_marg = tree_marginalize(
            self.likelihood.expected_statistics(),
            weights=assignment_probs,
            dims=self.like_mix_dims,
            keepdims=keepdims,
        )
        residual = (self.likelihood.residual * self.assignments.mean).sum(
            tuple(range(-self.assignments.event_dim, 0)), keepdims=keepdims
        )
        return self.likelihood.__class__(
            expectations=expected_stats_marg, residual=residual
        )

    def marginalize_nat_params(self, keepdims=False) -> ExponentialFamily:
        """This returns the marginalized natural parameters of the likelihood distribution `self.likelihood`, using the posterior assignment probabilities given by `self.assignments`."""

        assignment_probs = jnp.expand_dims(
            self.assignments.mean, axis=tuple(range(-self.event_dim, 0))
        )
        nat_params_marg = tree_marginalize(
            self.likelihood.nat_params,
            weights=assignment_probs,
            dims=self.like_mix_dims,
            keepdims=keepdims,
        )
        residual = (self.likelihood.residual * self.assignments.mean).sum(
            tuple(range(-self.assignments.event_dim, 0)), keepdims=keepdims
        )
        return self.likelihood.__class__(nat_params=nat_params_marg, residual=residual)

    @multimethod
    def __mul__(self, other: Delta) -> Delta:
        """
        Overloads the * operator for Mixture messages, which multiplies the two by first marginalizing out the assignment probabilities
        before doing a standard overloaded multiply on the likelihood instances (which will call the * operator on whatever ExponentialFamily instances are stored in the likelihood attribute of the Mixture instances)
        """
        return other.copy()

    @multimethod
    def __mul__(self, other: ExponentialFamily):
        """
        This does the VMP version of multiplication, which is different from the standard multiplication for mixture messages
        which marginalizes out the assignment probabilities before doing the multiplication.
        """

        assignment_dims = tuple(range(-self.assignments.event_dim, 0))
        q_x_z = self.likelihood * other.expand_batch_shape(assignment_dims)
        logits = (
            self.assignments.logits
            + self.likelihood.residual
            - self.likelihood.log_partition().squeeze((-1, -2))
            + q_x_z.log_partition().squeeze((-1, -2))
        )
        q_z_l = Multinomial(ArrayDict(logits=logits))

        return MixtureMessage(
            likelihood=q_x_z, assignments=q_z_l, average_type=self.average_type
        )

    @multimethod
    def __mul__(self, other):
        """
        Overloads the * operator for Mixture messages, which multiplies the two by first marginalizing out the assignment probabilities
        before doing a standard overloaded multiply on the likelihood instances (which will call the * operator on whatever ExponentialFamily instances are stored in the likelihood attribute of the Mixture instances)
        """

        marginalized_self = self.marginalize()
        marginalized_other = (
            other.marginalize() if isinstance(other, self.__class__) else other
        )

        if not isinstance(
            marginalized_other, marginalized_self.__class__
        ):  # Check if the other instance is of the same class as self
            raise ValueError(
                f"Cannot multiply {type(marginalized_self)} with {type(marginalized_other)}"
            )

        return marginalized_self * marginalized_other
