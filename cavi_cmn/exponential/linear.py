# This code is part of the VersesTech Repository `cavi-cmn` (https://github.com/VersesTech/cavi-cmn).
# It is licensed under the VERSES Academic Research License.
#
# For more information, please refer to the license file:
# https://github.com/VersesTech/cavi-cmn/blob/main/license.txt

from typing import Optional
from jaxtyping import Array
from jax import numpy as jnp

from cavi_cmn import ArrayDict
from cavi_cmn.distribution import Distribution
from cavi_cmn.utils import params_to_tx
from cavi_cmn.exponential import ExponentialFamily

DEFAULT_EVENT_DIM = 2


@params_to_tx(
    {
        "a_inv_sigma_a": "xx",
        "inv_sigma_a": "yx",
        "inv_sigma": "yy",
        "logdet_inv_sigma": "ones",
    }
)
class Linear(ExponentialFamily):
    """Linear mapping"""

    x_dim: int
    y_dim: int

    pytree_aux_fields = ("x_dim", "y_dim")

    def __init__(
        self,
        nat_params: ArrayDict,
        event_dim: Optional[int] = DEFAULT_EVENT_DIM,
        **parent_kwargs
    ):
        batch_shape, event_shape = self.infer_shapes(nat_params.inv_sigma_a, event_dim)
        self.x_dim, self.y_dim = event_shape[-1], event_shape[-2]
        super().__init__(
            DEFAULT_EVENT_DIM,
            batch_shape,
            event_shape,
            nat_params=nat_params,
            **parent_kwargs
        )
        self._validate_nat_params(nat_params)

    def statistics(self, data: tuple[Array]) -> ArrayDict:
        """
        Returns the sufficient statistics T(x): [xx, yx, yy, 1]
        """
        x, y = data

        xx = x * x.mT
        yy = y * y.mT
        yx = y * x.mT
        ones = jnp.broadcast_to(jnp.ones(1), xx.shape[:-2] + (1, 1))

        return ArrayDict(xx=xx, yx=yx, yy=yy, ones=ones)

    def log_measure(self, data: tuple[Array]) -> Array:
        """
        Returns the log of the measure of the exponential family.
        """
        return -0.5 * self.y_dim * jnp.log(2 * jnp.pi)

    def stats_from_probs(self, data: tuple[Distribution]) -> ArrayDict:
        x, y = data
        xx = x.expected_xx()
        yy = y.expected_xx()
        yx = y.mean * x.mean.mT
        ones = jnp.broadcast_to(jnp.ones(1), xx.shape[:-2] + (1, 1))

        return ArrayDict(xx=xx, yx=yx, yy=yy, ones=ones)
