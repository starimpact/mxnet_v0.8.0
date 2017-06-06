# pylint: skip-file
import mxnet as mx
import numpy as np


ori_shape = [(40, 12)]
ori_index = [[-3, -2, -1, 0, 4, 5, 7, 11, 15, 22]]
semipos = 3
shape = (10, 12)
keys = [1, 2, 4, 5, 7, 11]
def init_kv():
    """init kv """
    kv = mx.kv.create()
    # single
    kv.init_partial(3, mx.nd.zeros(shape), ori_shape, ori_index)
    # list
    klen = len(keys)
    kv.init_partial(keys, [mx.nd.zeros(shape)] * klen, ori_shape * klen, ori_index * klen)
    return kv


def check_diff_to_scalar(A, x):
    """ assert A == x"""
    assert(np.sum(np.abs(A - x)) == 0)

def test_single_kv_pair():
    """single key-value pair push & pull"""

    kv = init_kv()
    kv.push_partial(3, mx.nd.ones(shape), ori_shape, ori_index)
    val = mx.nd.empty(shape)
    kv.pull_partial(3, val, ori_shape, ori_index)
    val = val.asnumpy()
    check_diff_to_scalar(val[semipos:], 1)

def test_init():
    """test init"""
    kv = mx.kv.create()
    kv.init_partial(3, mx.nd.ones(shape)*4, ori_shape, ori_index)
    a = mx.nd.zeros(shape)
    kv.pull_partial(3, a, ori_shape, ori_index)
    a = a.asnumpy()[semipos:]
    check_diff_to_scalar(a, 4)

def test_list_kv_pair():
    """list key-value pair push & pull"""

    kv = init_kv()
    
    klen = len(keys)
    kv.push_partial(keys, [mx.nd.ones(shape)*4] * klen, ori_shape * klen, ori_index * klen)
    val = [mx.nd.empty(shape)] * klen
    kv.pull_partial(keys, val, ori_shape * klen, ori_index * klen)
    for v in val:
        v = v.asnumpy()
        check_diff_to_scalar(v[semipos:], 4)


def test_aggregator():
    """aggregate value on muliple devices"""

    kv = init_kv()

    # devices
    num_devs = 4
    devs = [mx.Context('cpu', i) for i in range(num_devs)]

    # single
    vals = [mx.nd.ones(shape, d) for d in devs]

    kv.push_partial(3, vals, ori_shape, ori_index)
    kv.pull_partial(3, vals, ori_shape, ori_index)

    for v in vals:
        v = v.asnumpy()[semipos:]
        check_diff_to_scalar(v, num_devs)

    # list
    klen = len(keys)
    vals = [[mx.nd.ones(shape, d)*2.0 for d in devs]] * klen
    kv.push_partial(keys, vals, ori_shape * klen, ori_index * klen)
    kv.pull_partial(keys, vals, ori_shape * klen, ori_index * klen)

    for vv in vals:
        for v in vv:
            v = v.asnumpy()[semipos:]
            check_diff_to_scalar(v, num_devs * 2.0)


def partial_updater(key, recv, local, state):
    """use updater: +="""
    local += recv

def test_updater(dev = 'cpu'):
    """updater"""

    kv = init_kv()
    kv._set_partial_updater(partial_updater)

    # devices
    num_devs = 4
    devs = [mx.Context(dev, i) for i in range(num_devs)]

    # single
    vals = [mx.nd.ones(shape, d) for d in devs]

    kv.push_partial(3, vals, ori_shape, ori_index)
    kv.pull_partial(3, vals, ori_shape, ori_index)

    for v in vals:
        v = v.asnumpy()[semipos:]
        check_diff_to_scalar(v, num_devs)

    # list
    klen = len(keys)
    vals = [[mx.nd.ones(shape, d) for d in devs]] * klen

    num_push = 4
    for i in range(num_push):
        kv.push_partial(keys, vals, ori_shape * klen, ori_index * klen)

    kv.pull_partial(keys, vals, ori_shape * klen, ori_index * klen)

    for vv in vals:
        for v in vv:
            v = v.asnumpy()[semipos:]
            check_diff_to_scalar(v, num_devs * num_push)

def test_get_type():
    kvtype = 'local_allreduce_cpu'
    kv = mx.kv.create(kvtype)
    assert kv.type == kvtype

if __name__ == '__main__':
    test_init()
    test_get_type()
    test_single_kv_pair()
    test_list_kv_pair()
    test_aggregator()
    test_updater()
