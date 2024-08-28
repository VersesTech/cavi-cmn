from collections.abc import Sequence
from typing import Union, Tuple, List, Optional
from jaxtyping import Array, PRNGKeyArray
from multimethod import multimethod
import warnings

import jax
from jax import lax
from jax import numpy as jnp
from jax.tree_util import tree_map, tree_reduce
from jax import random as jr

from cavi_cmn import ArrayDict, Distribution, Delta
from cavi_cmn.exponential import MixtureMessage, MultivariateNormal, Multinomial
from cavi_cmn.models import Model
from cavi_cmn.transforms import Transform, MultinomialRegression


DEFAULT_EVENT_DIM = 2


class Sequential(Model):
    """A sequence of `cavi_cmn.Distribution`, `cavi_cmn.Transform`, or `cavi_cmn.Model` objects
    that are connected to form a hierarchical generative model with the following factorized form

    .. math::
        p(Y | \Phi_{0:L}, z_{0:L}) = p(Y | z_L, \Phi_L) (\prod_{l=0}^{L-1} p(z_{l+1} | z_{l}, \Phi_l) P(\Phi_l)) p(z_0)

    where:
        - :math: Y are observations (potentially multi-sample and multi-variate)
        - :math: z_{0:L} are latent variables that are shared between the layer-wise models
        - :math: \Phi_{0:L} are layer-specific variables, and may include both parameters \theta_l and additional latent variables u_l
        - :math: p(z_0) is the prior distribution over the latent variables at the 0-th layer, which can also be clamped to observed data (e.g., in a supervised learning context)

    Note that the functional form of p(z_{l+1} | z_{l}, \Phi_l) is arbitrary and
    can be any Distribution, Transform, or Model class that has `forward`, `backward`, and `update_from_data()` or `update_from_probabilities()` methods.
    """

    pytree_data_fields = ("layers",)

    def __init__(self, layers: Sequence[Distribution]):
        """**Arguments:**

        - `layers`: A sequence of `cavi_cmn.Distribution`or `cavi_cmn.Transform` or `cavi_cmn.Model` objects that represent the layers of the model. Each layer can be thought of as a factor that parameterizes
        the conditional distribution p(z_{l+1} | z_{l}, \Phi_l) at the level of hierarchical generative model defining the network.
        """

        self.layers = tuple(layers)
        self.input_event_dim = self.layers[0].event_dim

        batch_shape = self.layers[0].batch_shape

        input_dim = self.layers[0].event_shape[-1]
        output_dim = self.layers[-1].event_shape[-self.layers[-1].event_dim]
        event_shape = (output_dim, input_dim)
        super().__init__(DEFAULT_EVENT_DIM, batch_shape, event_shape)

    def __call__(
        self,
        x: Union[Array, Distribution],
    ) -> Union[Array, Distribution]:
        """**Arguments:**

        - `x`: passed to the first member of the sequence.

        **Returns:**
        The output of the last member of the sequence.
        """

        return self.predict(x)

    @multimethod
    def predict(self, x: Array):
        """**Arguments:**

        - `x`: Input data (Array) that is passed to the first member of the sequence.

        **Returns:**
        The output of the last member of the sequence.
        """

        x = Delta(x, event_dim=self.layers[0].event_dim)
        return self.predict(x)

    @multimethod
    def predict(self, x: Distribution):
        """**Arguments:**

        - `x`: Input message (Distribution) that is passed to the first member of the sequence.

        **Returns:**
        The output of the last member of the sequence.
        """

        for layer in self.layers:
            x = layer.forward(x)
        return x

    @multimethod
    def _predict(self, x: Array, layers: tuple):
        """**Arguments:**

        - `x`: Input data (Array) that is passed to the first member of the sequence.
        - `layers`: Tuple of layers that represent the sequence of the model.

        **Returns:**
        The output of the last member of the sequence.
        """

        x = Delta(x, event_dim=self.layers[0].event_dim)

        return self._predict(x, layers)

    @multimethod
    def _predict(self, x: Distribution, layers: tuple):
        """**Arguments:**

        - `x`: Input message (Distribution) that is passed to the first member of the sequence.
        - `layers`: Tuple of layers that represent the sequence of the model.

        **Returns:**
        The output of the last member of the sequence.
        """

        for layer in layers:
            x = layer.forward(x)
        return x

    @multimethod
    def forward(self, x: Array):
        """**Arguments:**

        - `x`: Input data (Array) that is passed to the first member of the sequence.

        **Returns:**
        List of forward messages of each layer in the model, including the input message `x`.
        """

        x = Delta(x, event_dim=self.layers[0].event_dim)

        return self.forward(x)

    @multimethod
    def forward(self, x: Distribution):
        """**Arguments:**

        - `x`: Input message (Distribution) that is passed to the first member of the sequence.

        **Returns:**
        List of forward messages of each layer in the model, including the input message `x`.
        """

        forward_msgs = [x] + [None] * len(self)
        for i, layer in enumerate(self.layers, start=1):
            forward_msgs[i] = layer.forward(forward_msgs[i - 1])
        return forward_msgs

    @multimethod
    def _forward(self, x: Array, layers: tuple):
        """**Arguments:**

        - `x`: Input data (Array) that is passed to the first member of the sequence.
        - `layers`: Tuple of layers that represent the sequence of the model.

        **Returns:**
        List of forward messages of each layer in the model, including the input message `x`.
        """

        x = Delta(x, event_dim=self.layers[0].event_dim)

        return self._forward(x, layers)

    @multimethod
    def _forward(self, x: Distribution, layers: tuple):
        """**Arguments:**

        - `x`: Input message (Distribution) that is passed to the first member of the sequence.
        - `layers`: Tuple of layers that represent the sequence of the model.

        **Returns:**
        List of forward messages of each layer in the model, including the input message `x`.
        """

        forward_msgs = [x] + [None] * len(self)
        for i, layer in enumerate(layers, start=1):
            forward_msgs[i] = layer.forward(forward_msgs[i - 1])
        return forward_msgs

    @multimethod
    def backward(self, y: Array):
        """**Arguments:**

        - `Y`: Output message (Distribution) that is passed to the last member of the sequence.

        **Returns:**
        List of backward messages for between each pair of layers in the sequence. In general, the length of the list is `len(self) + 1`
        and element `i` of the sequence is the backward message that is on the output side of layer `i-1` (or simply the edge facing the input data)
        and the input side of layer `i` (or simply the edge facing the output data).
        """

        y = Delta(y, event_dim=self.layers[-1].event_dim)
        return self.backward(y)

    @multimethod
    def backward(self, y: Distribution):
        """**Arguments:**

        - `Y`: Output message (Distribution) that is passed to the last member of the sequence.

        **Returns:**
        List of backward messages for between each pair of layers in the sequence. In general, the length of the list is `len(self) + 1`
        and element `i` of the sequence is the backward message that is on the output side of layer `i-1` (or simply the edge facing the input data)
        and the input side of layer `i` (or simply the edge facing the output data).
        """

        backward_msgs = [None] * len(self) + [y]
        for i, layer in reversed(list(enumerate(self.layers))):
            backward_msgs[i] = layer.backward(backward_msgs[i + 1])
        return backward_msgs

    @multimethod
    def _backward(self, y: Array, layers: tuple):
        """**Arguments:**

        - `Y`: Output message (Distribution) that is passed to the last member of the sequence.

        **Returns:**
        List of backward messages for between each pair of layers in the sequence. In general, the length of the list is `len(self) + 1`
        and element `i` of the sequence is the backward message that is on the output side of layer `i-1` (or simply the edge facing the input data)
        and the input side of layer `i` (or simply the edge facing the output data).
        """

        y = Delta(y, event_dim=self.layers[-1].event_dim)
        return self._backward(y, layers)

    @multimethod
    def _backward(self, y: Distribution, layers: tuple):
        """**Arguments:**

        - `Y`: Output message (Distribution) that is passed to the last member of the sequence.

        **Returns:**
        List of backward messages for between each pair of layers in the sequence. In general, the length of the list is `len(self) + 1`
        and element `i` of the sequence is the backward message that is on the output side of layer `i-1` (or simply the edge facing the input data)
        and the input side of layer `i` (or simply the edge facing the output data).
        """

        backward_msgs = [None] * len(self) + [y]
        for i, layer in reversed(list(enumerate(layers))):
            backward_msgs[i] = layer.backward(backward_msgs[i + 1])
        return backward_msgs

    @multimethod
    def forward_backward_pass(self, x: Array, y: Array):
        """
        This function calls the functional private method `_forward_backward_pass` that computes the marginals of the model, using the current
        settings of the layers' parameters.

        **Arguments:**
        - `x`: input data (Array) that is passed to the first member of the sequence.
        - `y`: output data (Array) that is passed to the last member of the sequence.

        **Returns:**
        A list of messages that represent the marginals of the model, where the first element is the marginal of the input message `x` and the last element is the marginal of the output message `y`.
        """
        forward_msgs = self.forward(x)
        backward_msgs = self.backward(y)

        return self._multiply_messages(forward_msgs, backward_msgs)

    @multimethod
    def forward_backward_pass(self, x: Distribution, y: Distribution):
        """
        This function calls the functional private method `_forward_backward_pass` that computes the marginals of the model, using the current
        settings of the layers' parameters.

        **Arguments:**
        - `x`: input message (Distribution) that is passed to the first member of the sequence.
        - `y`: output message (Distribution) that is passed to the last member of the sequence.

        **Returns:**
        A list of messages that represent the marginals of the model, where the first element is the marginal of the input message `x` and the last element is the marginal of the output message `y`.
        """
        return self._forward_backward_pass(x, y, self.layers)

    def _forward_backward_pass(
        self,
        x: Union[Array, Distribution],
        y: Union[Array, Distribution],
        layers: tuple,
    ):
        """
        The functional private method that computes the marginals of the model, using the current settings of the layers' parameters.
        **Arguments:**
        - `x`: input data or message (Array or Distribution) that is passed to the first member of the sequence.
        - `y`: output data or message (Array or Distribution) that is passed to the last member of the sequence.
        """
        forward_msgs = self._forward(x, layers)
        backward_msgs = self._backward(y, layers)
        return self._multiply_messages(forward_msgs, backward_msgs)

    def _multiply_messages(self, forward_msgs: list, backward_msgs: list):
        """
        This computes the marginals of the model, using the forward and backward messages computed by the `_forward` and `_backward` methods.
        **Arguments:**
        - `forward_msgs`: list of forward messages computed by the `_forward` method.
        - `backward_msgs`: list of backward messages computed by the `_backward` method.

        **Returns:**
        A list of messages that represent the marginals of the model, where the first element is the marginal of the input message `x` and the last element is the marginal of the output message `y`.
        """

        return tree_map(
            lambda x, y: x * y,
            forward_msgs,
            backward_msgs,
            is_leaf=lambda x: isinstance(x, Distribution),
        )

    def fit_vmp_for_loop(
        self,
        key,
        x: Union[Array, Distribution],
        y: Union[Array, Distribution],
        edge_init_method: str = "forward",  # 'random', 'forward', 'product'
        n_m_steps: int = 5,
        n_e_steps: int = 1,
    ):
        """
        For-loop, non-JITTed version of the fit_vmp method. This method fits the model to data using the Variational Message Passing (VMP) algorithm. The VMP algorithm is a message-passing algorithm that
        iteratively updates the latents and parameters of the model by maximizing the Evidence Lower Bound (ELBO) of the model, using a variational approximation to
        the latent variables (x, z) and the model parameters \Phi. Note that in this form of VMP, we assume that the posterior over each edge's latents
        are a joint distribution over the continuous and discrete latents, i.e., q(x, z) = q(x|z)q(z). This is the so-called "conditional VB" approach to VMP.
        """

        ## initalize posteriors using a forward / backward pass
        forward_msgs = self._forward(x, self.layers)
        backward_msgs = self._backward(y, self.layers)

        message_keys = jr.split(key, num=len(self) - 1)

        if edge_init_method == "product":
            edge_posteriors = self._multiply_messages(forward_msgs, backward_msgs)
        else:
            edge_posteriors = self.initialize_posteriors(
                message_keys, forward_msgs, backward_msgs, init_method=edge_init_method
            )

        message_keys = jr.split(message_keys[-1], num=len(self) - 1)

        if len(self) > 2 and n_e_steps == 1:
            # show a warning here that the number of e-steps is set to 1, but the number of layers is greater than 2
            warnings.warn(
                "The number of e-steps is set to 1, but the number of layers is greater than 2. This may lead to suboptimal results."
            )
            # n_e_steps = 2*(len(self)-1)

        for _ in range(n_m_steps):
            """Initialize the forwards and backwards messages"""

            ## initalize posteriors using a forward / backward pass
            forward_msgs = self._forward(x, self.layers)
            backward_msgs = self._backward(y, self.layers)

            if edge_init_method == "product":
                edge_posteriors = self._multiply_messages(forward_msgs, backward_msgs)
            else:
                edge_posteriors = self.initialize_posteriors(
                    message_keys,
                    forward_msgs,
                    backward_msgs,
                    init_method=edge_init_method,
                )

            message_keys = jr.split(message_keys[-1], num=len(self) - 1)

            """ First update the latents (the E steps) """
            for _ in range(n_e_steps):

                edge_posteriors = self.vmp_update(edge_posteriors)

            """ Now update the parameters (the M steps) """
            for j, layer in enumerate(self.layers, start=1):

                layer.update_from_probabilities(
                    (edge_posteriors[j - 1], edge_posteriors[j])
                )

        return edge_posteriors

    def fit_vmp(
        self,
        key,
        x: Union[Array, Distribution],
        y: Union[Array, Distribution],
        edge_init_method: str = "forward",  # 'random', 'forward', 'product'
        n_m_steps: int = 5,
        n_e_steps: int = 1,
    ):
        """
        Method that fits the model to data using the Variational Message Passing (VMP) algorithm. The VMP algorithm is a message-passing algorithm that
        iteratively updates the latents and parameters of the model by maximizing the Evidence Lower Bound (ELBO) of the model, using a variational approximation to
        the latent variables (x, z) and the model parameters \Phi. Note that in this form of VMP, we assume that the posterior over each edge's latents
        are a joint distribution over the continuous and discrete latents, i.e., q(x, z) = q(x|z)q(z). This is the so-called "conditional VB" approach to VMP.
        """

        ## initalize posteriors using a forward / backward pass
        forward_msgs = self._forward(x, self.layers)
        backward_msgs = self._backward(y, self.layers)

        message_keys = jr.split(key, num=len(self) - 1)

        if edge_init_method == "product":
            initial_posteriors = self._multiply_messages(forward_msgs, backward_msgs)
        else:
            initial_posteriors = self.initialize_posteriors(
                message_keys, forward_msgs, backward_msgs, init_method=edge_init_method
            )

        message_keys = jr.split(message_keys[-1], num=len(self) - 1)

        if len(self) > 2 and n_e_steps == 1:
            # show a warning here that the number of e-steps is set to 1, but the number of layers is greater than 2
            warnings.warn(
                "The number of e-steps is set to 1, but the number of layers is greater than 2. This may lead to suboptimal results."
            )
            # n_e_steps = 2*(len(self)-1)

        def vb_iter_step(carry, i):

            keys, layers = carry

            ## initalize posteriors using a forward / backward pass
            forward_msgs = self._forward(x, layers)
            backward_msgs = self._backward(y, layers)

            if edge_init_method == "product":
                edge_posteriors = self._multiply_messages(forward_msgs, backward_msgs)
            else:
                edge_posteriors = self.initialize_posteriors(
                    keys, forward_msgs, backward_msgs, init_method=edge_init_method
                )

            keys = jr.split(keys[-1], num=len(layers) - 1)

            """ First update the latents (the E steps) """

            for _ in range(n_e_steps):

                edge_posteriors = self._vmp_update(layers, edge_posteriors)

            """ Now update the parameters (the M steps) """
            for j, layer in enumerate(layers, start=1):
                layer.update_from_probabilities(
                    (edge_posteriors[j - 1], edge_posteriors[j])
                )

            return (keys, layers), (edge_posteriors)

        initial_carry = (message_keys, self.layers)
        (keys, updated_layers), posteriors_over_m_steps = lax.scan(
            vb_iter_step, initial_carry, jnp.arange(n_m_steps)
        )

        self.layers = updated_layers
        return posteriors_over_m_steps

    def vmp_update(self, edges):
        """
        This function computes the VMP update for the latents of the model, given the current settings of the layers' parameters.
        """

        for edge_i in range(1, len(self)):
            q_xl_given_z = self.layers[edge_i - 1].forward(edges[edge_i - 1])
            ll_xl = self.layers[edge_i].backward(edges[edge_i + 1])
            edges[edge_i] = q_xl_given_z * ll_xl

        return edges

    def _vmp_update(self, layers, edges):
        """
        This function computes the VMP update for the latents of the model, given the current settings of the layers' parameters.
        This is the functional version of the `vmp_update` method.
        """

        for edge_i in range(1, len(layers)):
            q_xl_given_z = layers[edge_i - 1].forward(edges[edge_i - 1])
            ll_xl = layers[edge_i].backward(edges[edge_i + 1])
            edges[edge_i] = q_xl_given_z * ll_xl

        return edges

    def initialize_posteriors(
        self, message_keys, forward_msgs, backward_msgs, init_method="random"
    ):

        q_edges = [None] * (len(self) + 1)

        # Now multiply all the forwards and backards messages
        q_edges[0] = forward_msgs[0] * backward_msgs[0]
        q_edges[-1] = forward_msgs[-1] * backward_msgs[-1]

        # in the case of conditional VB, in the intermediate layers (layers with indices 1 to len(self)-2) are mixture messages
        for i in range(1, len(self)):
            if init_method == "forward":
                q_edges[i] = forward_msgs[
                    i
                ]  # initialize the edges to the outputs of the forwards pass
            elif init_method == "random":
                p_y_given_z = forward_msgs[i].likelihood
                p_z = forward_msgs[i].assignments
                q_x_given_z = p_y_given_z.__class__(
                    batch_shape=p_y_given_z.batch_shape,
                    event_shape=p_y_given_z.event_shape,
                    init_key=message_keys[i - 1],
                )
                q_z = p_z.__class__(
                    batch_shape=p_z.batch_shape, event_shape=p_z.event_shape
                )
                q_edges[i] = MixtureMessage(likelihood=q_x_given_z, assignments=q_z)

        return q_edges

    def fit(
        self,
        x: Union[Array, Distribution],
        y: Union[Array, Distribution],
        n_iters: int = 2,
    ):
        """
        WARNING: This method has not been tested with latest versions of ConditionalMixture and MultinomialRegression (since emphasis has
        shifted to working on variational message passing, as oppposed to standard forward/backward), so it
        may be deprecated and is not guaranteed to work.
        """

        def step_fn(carry, i):
            layers = carry
            marginals = self._forward_backward_pass(x, y, layers)

            elbos_per_layer = []  # list of len(layers)
            for j, layer in enumerate(layers, start=1):
                elbo = layer.update_from_probabilities((marginals[j - 1], marginals[j]))
                elbos_per_layer.append(elbo)

            # when each marginal is an instance of a message class, then we can replace this solution with
            marginal_entropies = tree_reduce(
                lambda entr_i, entr_j: entr_i + entr_j,
                (
                    tree_map(
                        lambda x: x.entropy(),
                        marginals,
                        is_leaf=lambda leaf: isinstance(leaf, Distribution),
                    )
                ),
            )
            # just a scalar, \sum_i H[Q(i)]
            # sum(elbos_per_layer) + marginal_entropies

            return layers, (elbos_per_layer, marginal_entropies, marginals)

        initial_carry = self.layers
        updated_layers, (elbos_per_layer, marginal_entropies, marginals) = lax.scan(
            step_fn, initial_carry, jnp.arange(n_iters)
        )

        self.layers = updated_layers
        return elbos_per_layer, marginal_entropies, marginals

    def fit_for_loop(
        self,
        x: Union[Array, Distribution],
        y: Union[Array, Distribution],
        n_iters: int = 2,
    ):
        """
        WARNING: This method has not been tested with latest versions of ConditionalMixture and MultinomialRegression (since emphasis has
        shifted to working on variational message passing, as oppposed to standard forward/backward), so it
        may be deprecated and is not guaranteed to work.
        """

        elbos_over_iters = []
        marginal_entropies_over_iters = []
        for i in range(n_iters):
            marginals = self.forward_backward_pass(x, y)  # compute a list of marginals

            for i, marg in enumerate(marginals):

                if type(marg) == MultivariateNormal:
                    # check if the marginal has a well-behaved precision matrix
                    eigs = jnp.linalg.eigh(marg.expected_xx())[0]
                    print(jnp.any(eigs <= 0))

            elbos_per_layer = []
            for j, layer in enumerate(self.layers, start=1):

                elbo = layer.update_from_probabilities((marginals[j - 1], marginals[j]))
                elbos_per_layer.append(elbo)

            elbos_over_iters.append(elbos_per_layer)

            # when each marginal is an instance of a message class, then we can replace this solution with
            marginal_entropies = tree_reduce(
                lambda entr_i, entr_j: entr_i + entr_j,
                (
                    tree_map(
                        lambda x: x.entropy(),
                        marginals,
                        is_leaf=lambda leaf: isinstance(leaf, Distribution),
                    )
                ),
            )
            marginal_entropies_over_iters.append(marginal_entropies)

        return elbos_over_iters, marginal_entropies_over_iters

    def fit_fbi(
        self,
        x: Union[Array, Distribution],
        y: Union[Array, Distribution],
        n_iters: int = 2,
    ):
        """
        WARNING: This method has not been tested with latest versions of ConditionalMixture and MultinomialRegression (since emphasis has
        shifted to working on variational message passing, as oppposed to standard forward/backward), so it
        may be deprecated and is not guaranteed to work.
        """

        if isinstance(y, Array):
            y = Delta(y, event_dim=self.layers[-1].event_dim)

        def step_fn(carry, i):
            layers = carry
            forward_msgs = self._forward(x, layers)

            backward_msgs = [None] * len(self) + [y]
            marginals = [None] * len(self) + [y]

            elbos_per_layer = []
            # fbi algorithm starts at the output side of the model and passes backwards messages, updating layer-wise parameters at each go
            for j, layer in reversed(list(enumerate(layers))):
                backward_msg_preupdate = layer.backward(backward_msgs[j + 1])
                marginal_j_preupdate = forward_msgs[j] * backward_msg_preupdate
                elbo = layer.update_from_probabilities(
                    (marginal_j_preupdate, marginals[j + 1])
                )
                elbos_per_layer.append(elbo)
                backward_msgs[j] = layer.backward(backward_msgs[j + 1])
                marginals[j] = forward_msgs[j] * backward_msgs[j]

            # when each marginal is an instance of a message class, then we can replace this solution with
            marginal_entropies = tree_reduce(
                lambda entr_i, entr_j: entr_i + entr_j,
                (
                    tree_map(
                        lambda x: x.entropy(),
                        marginals,
                        is_leaf=lambda leaf: isinstance(leaf, Distribution),
                    )
                ),
            )

            return layers, (elbos_per_layer, marginal_entropies, marginals)

        initial_carry = self.layers
        updated_layers, (elbos_per_layer, marginal_entropies, marginals) = lax.scan(
            step_fn, initial_carry, jnp.arange(n_iters)
        )

        self.layers = updated_layers
        return elbos_per_layer, marginal_entropies, marginals

    def fit_fbi_for_loop(
        self,
        x: Union[Array, Distribution],
        y: Union[Array, Distribution],
        n_iters: int = 2,
    ):
        """
        WARNING: This method has not been tested with latest versions of ConditionalMixture and MultinomialRegression (since emphasis has
        shifted to working on variational message passing, as oppposed to standard forward/backward), so it
        may be deprecated and is not guaranteed to work.
        """

        if isinstance(y, Array):
            y = Delta(y, event_dim=self.layers[-1].event_dim)

        for i in range(n_iters):

            forward_msgs = self._forward(x, self.layers)

            backward_msgs = [None] * len(self) + [y]
            marginals = [None] * len(self) + [y]

            elbos_per_layer = []

            for j, layer in reversed(list(enumerate(self.layers))):
                backward_msg_preupdate = layer.backward(backward_msgs[j + 1])
                marginal_j_preupdate = forward_msgs[j] * backward_msg_preupdate
                elbo = layer.update_from_probabilities(
                    (marginal_j_preupdate, marginals[j + 1])
                )
                elbos_per_layer.append(elbo)
                backward_msgs[j] = layer.backward(backward_msgs[j + 1])
                marginals[j] = forward_msgs[j] * backward_msgs[j]

            marginal_entropies = tree_reduce(
                lambda entr_i, entr_j: entr_i + entr_j,
                (
                    tree_map(
                        lambda x: x.entropy(),
                        marginals,
                        is_leaf=lambda leaf: isinstance(leaf, Distribution),
                    )
                ),
            )

        return elbos_per_layer, marginal_entropies

    def __getitem__(self, i: Union[int, slice]) -> Distribution:
        if isinstance(i, int):
            return self.layers[i]
        elif isinstance(i, slice):
            return Sequential(self.layers[i])
        else:
            raise TypeError(f"Indexing with type {type(i)} is not supported")

    def __iter__(self):
        yield from self.layers

    def __len__(self):
        return len(self.layers)


class ConditionalMixtureNetwork(Sequential):
    """This class implements variational message passing
    for the following generative model, the so-called "Single hidden layer Directed Mixture Network":

    p(y|x_1)p(x_1|x_0, z_1)p(z_1|x_0)

    where:
    - x_1, x_0 are continuous
    - y, z_1 are discrete.

    - p(y|x_1) is a MultinomialLogisticRegression
    - p(x_1|x_0, z_0) is a Mixture of Linear Transforms
    - p(z_0|x_0) is a MultinomialLogisticRegression
    and therefore the combined distribution p(x_1|x_0, z_0)p(z_0|x_0) is a ConditionalMixture.

    The VMP algorithm assumes an approximate posterior with the so-called 'conditional form', i.e., q(x_1, z_1) = q(x_1|z_1)q(z_1).
    """

    pytree_aux_fields = ("n_backwards_iters", "backwards_type", "compute_elbo")

    # write an __init__() method that just adds two arguments called "n_backwards_iters" and "backwards_type"
    # and then handles rest of the initialization in the super().__init__() call
    def __init__(
        self,
        layers: Sequence[Distribution],
        n_backwards_iters: int = 2,
        backwards_type: str = "smooth",
        compute_elbo=True,
    ):
        super().__init__(layers)
        self.n_backwards_iters = n_backwards_iters
        self.backwards_type = backwards_type
        self.compute_elbo = compute_elbo

    @multimethod
    def _backward_smooth(self, y: Array, forward_messages, layers):
        y_delta = Delta(y, event_dim=layers[-1].event_dim)
        return self._backward_smooth(y_delta, forward_messages, layers)

    @multimethod
    def _backward_smooth(self, y: Delta, forward_msgs, layers):
        q_edges = [None] * (len(layers) + 1)
        q_edges[0] = forward_msgs[0]
        q_edges[-1] = y

        for i, layer in reversed(list(enumerate(layers))):

            if isinstance(layer, MultinomialRegression):

                # get the forward message (a MixtureMessage from the previous ConditionalMixture's forward() call)
                q_x_z_forward, q_z_fwd = (
                    forward_msgs[i].likelihood,
                    forward_msgs[i].assignments,
                )

                # print(f'Shape of posterior q(y) at edge {i+1}: {q_edges[i+1].shape}')

                # compute q(x_{l}, z_{l}), where l refers to the index of the current layer (i)
                if self.backwards_type == "expand_around_unitnormal":
                    q_x_z_backward = layer.backward(
                        q_edges[i + 1], iters=self.n_backwards_iters
                    )
                    q_x_z_smoothed = q_x_z_forward * q_x_z_backward.expand_batch_shape(
                        -1
                    )

                elif self.backwards_type == "smooth":
                    # treat the extra batch dimension corresponding to the z variable as another sample dimension (to be treated in parallel)
                    # by doing q_x_forward.moveaxis(-1, 1) and adding a trivial extra sample dimension (i.e, batch shape) to q_edges[i + 1]
                    q_x_z_smoothed = layer.backward_smooth(
                        pY=q_edges[i + 1].expand_batch_shape(1),
                        pX=q_x_z_forward.moveaxis(-1, 1),
                        iters=self.n_backwards_iters,
                    ).moveaxis(1, -1)

                # print(f'Shape of q_x_z_smoothed: {q_x_z_smoothed.shape}')

                logits = (
                    q_z_fwd.logits + q_x_z_forward.residual + q_x_z_smoothed.residual
                )

                # create a new Multinomial distribution with the updated logits
                q_z_l = Multinomial(ArrayDict(logits=logits))

                # update the latent q(x_l, z_l) with the smoothed q(x_l, z_l) and the updated q(z_l)
                q_edges[i] = MixtureMessage(
                    likelihood=q_x_z_smoothed, assignments=q_z_l
                )

                elbo_contribs = (
                    q_z_l.log_normalizer.squeeze(-1) if self.compute_elbo else None
                )

            else:
                # print(f'Shape of posterior q(x_l|z_l) at edge {i+1}: {q_edges[i+1].likelihood.shape}')
                # print(f'Shape of posterior q(z_l) at edge {i+1}: {q_edges[i+1].assignments.shape}')
                backward_msg = layer.backward(q_edges[i + 1])
                # print(f"Shape of backward message from layer {i} of instance {type(layer)}: {backward_msg.shape}")
                q_edges[i] = forward_msgs[i] * backward_msg

        return q_edges, elbo_contribs

    def fit_vmp_for_loop(
        self,
        x: Union[Array, Distribution],
        y: Union[Array, Distribution],
        n_m_steps: int = 5,
        n_e_steps: int = 1,
    ):
        """
        For-loop, non-JITTed version of the fit_vmp method. This method fits the model to data using the Variational Message Passing (VMP) algorithm. The VMP algorithm is a message-passing algorithm that
        iteratively updates the latents and parameters of the model by maximizing the Evidence Lower Bound (ELBO) of the model, using a variational approximation to
        the latent variables (x, z) and the model parameters \Phi. Note that in this form of VMP, we assume that the posterior over each edge's latents
        are a joint distribution over the continuous and discrete latents, i.e., q(x, z) = q(x|z)q(z). This is the so-called "conditional VB" approach to VMP.
        """

        """ Do an initial E-step using a forward pass and then a smoothing pass"""
        forward_msgs = self._forward(x, self.layers)
        q_edges, elbo_contribs = self._backward_smooth(y, forward_msgs, self.layers)

        elbo = [None] * n_m_steps
        for m_i in range(n_m_steps):
            """Update the parameters (the M steps)"""
            for j, layer in enumerate(self.layers, start=1):

                layer.update_from_probabilities((q_edges[j - 1], q_edges[j]))

            """ Do another E-step using a forward pass and then a smoothing pass"""
            forward_msgs = self._forward(x, self.layers)
            q_edges, elbo_contribs = self._backward_smooth(y, forward_msgs, self.layers)

            elbo[m_i] = elbo_contribs.sum(0) - sum(
                [layer.kl_divergence() for layer in self.layers]
            )

        return elbo

    def fit_vmp(
        self,
        x: Union[Array, Distribution],
        y: Union[Array, Distribution],
        n_m_steps: int = 5,
        n_e_steps: int = 1,
        compute_accuracy: bool = False,  # flag for whether to compute test_acccuracy during training at each M step
        x_test: Optional[
            Union[Array, Distribution]
        ] = None,  # test data (assumed to be vector formatted)
        y_test: Optional[
            Union[Array, Distribution]
        ] = None,  # test labels (assumed to be integers)
    ):
        """
        Method that fits the model to data using the Variational Message Passing (VMP) algorithm. The VMP algorithm is a message-passing algorithm that
        iteratively updates the latents and parameters of the model by maximizing the Evidence Lower Bound (ELBO) of the model, using a variational approximation to
        the latent variables (x, z) and the model parameters \Phi. Note that in this form of VMP, we assume that the posterior over each edge's latents
        are a joint distribution over the continuous and discrete latents, i.e., q(x, z) = q(x|z)q(z). This is the so-called "conditional VB" approach to VMP.
        """

        def step_fn(carry, i):
            layers = carry

            forward_msgs = self._forward(x, layers)
            if compute_accuracy:
                predicted_labels = self._predict(x, layers).logits.argmax(-1)
                train_acc_i = jnp.mean(predicted_labels == y.argmax(-1), axis=0)

                predicted_labels = self._predict(x_test, layers).logits.argmax(-1)
                test_acc_i = jnp.mean(predicted_labels == y_test, axis=0)
            else:
                train_acc_i, test_acc_i = 0.0, 0.0

            q_edges, elbo_contribs = self._backward_smooth(y, forward_msgs, layers)

            if self.compute_elbo:
                elbo_i = elbo_contribs.sum(0) - sum(
                    [layer.kl_divergence() for layer in layers]
                )
            else:
                elbo_i = 0.0

            for j, layer in enumerate(layers, start=1):

                layer.update_from_probabilities((q_edges[j - 1], q_edges[j]))

            return layers, (elbo_i, train_acc_i, test_acc_i)

        updated_layers, (elbo, train_accuracy, test_accuracy) = lax.scan(
            step_fn, self.layers, jnp.arange(n_m_steps)
        )

        self.layers = updated_layers
        return elbo, train_accuracy, test_accuracy
