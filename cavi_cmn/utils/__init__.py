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
