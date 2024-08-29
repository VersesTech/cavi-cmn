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

"""
Transforms are distributions that are conditioned on latents.  They model p(y|x,\theta)
where y is the event, x is the input_event, and \theta gives the parameters.  When x and y 
are observed, the transforms expects:

x.shape = sample_shape + batch_shape + input_event_shape
y.shape = sample_shape + batch_shape + event_shape

where batch_shape refers to the batch shape of the parameters of the transform.  Note that 
technically the batch dimensions of x and y could include (1,)'s such as when you are doing a 
mixture of linear transforms or performing a batch of optimizations in parallel. But no matter what 
the batch_shape of x and y should allow for broadcasting.  

Transforms also have forward and backward routines that emit message distributions.  
"""

from cavi_cmn import Conjugate


class Transform(Conjugate):
    pass
