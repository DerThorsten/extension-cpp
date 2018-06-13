from __future__ import division
from __future__ import print_function

import argparse
import torch
import skimage.data
import vigra
import numpy


from torch.autograd import Variable, gradcheck

parser = argparse.ArgumentParser()
#parser.add_argument('example', choices=['py', 'cpp', 'cuda'], default='cpp')
parser.add_argument('-b', '--batch-size', type=int, default=3)
parser.add_argument('-f', '--features', type=int, default=17)
parser.add_argument('-s', '--state-size', type=int, default=5)
parser.add_argument('-c', '--cuda', action='store_true')
options = parser.parse_args()
options.example = 'cpp'

from cpp.spacc import *

shape = [2,2]
n_channels = 1



# image
X = torch.randn(1,n_channels, shape[0], shape[1])
sp = numpy.array([[1,1],[1,1]]).astype('int')
sp = numpy.require(sp, requirements=['C'])
print(sp.shape)
sp = torch.tensor(sp, dtype=torch.int)




variables = (
    Variable(X.double(), requires_grad=True),
    Variable(sp, requires_grad=False)
)

module = SpAccAvModule()

if gradcheck(SpAccAvFunction.apply, variables):
    print('Ok')
