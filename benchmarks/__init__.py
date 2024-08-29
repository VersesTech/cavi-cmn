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

from .mixture_of_experts import (
    fit_cmn_maximum_likelihood,
    multilayer_cmn_mle,
    compute_log_likelihood_cmn,
)
from .conditional_mixture_network import (
    fit_cmn_bbvi,
    fit_cmn_nuts,
    multilayer_cmn,
    compute_log_likelihood_bbvi,
    compute_log_likelihood_nuts,
)

from .utils import check_convergence_expfit, grid_of_points, plot_dataset
from .data import create_uci_dataloader, find_uci_stats, create_pinwheel_generator
