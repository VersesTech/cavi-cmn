# This code is part of the VersesTech Repository `cavi-cmn` (https://github.com/VersesTech/cavi-cmn).
# It is licensed under the VERSES Academic Research License.
#
# For more information, please refer to the license file:
# https://github.com/VersesTech/cavi-cmn/blob/main/license.txt

from .pytree import (
    ArrayDict,
    apply_add,
    apply_scale,
    params_to_tx,
    map_and_multiply,
    zeros_like,
    map_dict_names,
    size,
    tree_copy,
    sum_pytrees,
    tree_marginalize,
    tree_equal,
    tree_at,
)

from .math import (
    mvgammaln,
    mvdigamma,
    stable_logsumexp,
    stable_softmax,
    assign_unused,
    bdot,
    symmetrise,
    positive_leading_eigenvalues,
    make_posdef,
    inv_and_logdet,
)
from .input_handling import get_default_args, check_optim_args
