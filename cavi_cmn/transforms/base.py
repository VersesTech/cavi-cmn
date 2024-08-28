# This code is part of the VersesTech Repository `cavi-cmn` (https://github.com/VersesTech/cavi-cmn).
# It is licensed under the VERSES Academic Research License.
#
# For more information, please refer to the license file:
# https://github.com/VersesTech/cavi-cmn/blob/main/license.txt

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
