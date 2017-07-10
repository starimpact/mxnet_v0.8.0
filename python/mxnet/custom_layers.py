# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 14:22:25 2016

@author: mingzhang
"""

import os
# MXNET_CPU_WORKER_NTHREADS must be greater than 1 for custom op to work on CPU
os.environ["MXNET_CPU_WORKER_NTHREADS"] = "8"
from . import operator
import numpy as np


###############################################################################

class TakeOneDim0(operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        x = in_data[0]
        idx = int(in_data[1].asnumpy()[0])
        self.assign(out_data[0], req[0], x[idx])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        idx = int(in_data[1].asnumpy()[0])
        self.assign(in_grad[0][idx], req[0], out_grad[0])

@operator.register("TakeOneDim0")
class TakeOneDim0_Prop(operator.CustomOpProp):
    def __init__(self):
        super(TakeOneDim0_Prop, self).__init__(need_top_grad=True)
    
    def list_arguments(self):
        return ['data', 'index']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, inshape):
        data_shape = inshape[0]
        index_shape = (1,)
        output_shape = (data_shape[1],)
        return [data_shape, index_shape], [output_shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return TakeOneDim0()

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return [in_data[0], in_data[1], out_grad[0]]





