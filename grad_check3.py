from __future__ import division
from __future__ import print_function

import argparse
import torch
import skimage.data
import vigra
import numpy


from torch.autograd import Variable, gradcheck



from cpp.spacc import *

shape = [2,2]
n_channels = 1

sp = numpy.array([[1,1],[1,2]]).astype('int')
sp = numpy.require(sp, requirements=['C'])
labelscount = numpy.array([3,1]).astype('int')

# image
X = torch.randn(1,n_channels, shape[0], shape[1])
sp = torch.tensor(sp, dtype=torch.int)
labelscount = torch.tensor(labelscount, dtype=torch.int)



variables = (
    Variable(X.double(),   requires_grad=True),
    Variable(sp,          requires_grad=False),
    Variable(labelscount, requires_grad=False),
)


if gradcheck(LabelStatsAccumulatorFunction.apply, variables):
    print('Ok')
