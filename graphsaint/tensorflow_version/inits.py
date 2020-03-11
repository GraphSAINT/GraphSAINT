import tensorflow as tf
from graphsaint.globals import *
import numpy as np

# DISCLAIMER:
# Parts of this code file are derived from
# https://github.com/tkipf/gcn
# which is under an identical MIT license as graphsaint

"""
Initialization of weight matrices.
"""

def uniform(shape, scale=0.05, name=None):
    """Uniform init."""
    initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=DTYPE)
    return tf.Variable(initial, name=name)


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=DTYPE)
    return tf.Variable(initial, name=name)

def xavier(shape, name=None):
    init_dev = np.sqrt(2.0/(shape[0]+shape[1]))
    initial = tf.random_normal(shape,mean=0,stddev=init_dev,dtype=DTYPE)
    return tf.Variable(initial, name=name)


def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=DTYPE)
    return tf.Variable(initial, name=name)

def ones(shape, name=None):
    """All ones."""
    initial = tf.ones(shape, dtype=DTYPE)
    return tf.Variable(initial, name=name)

def trained(val_array, name=None):
    initial = tf.convert_to_tensor(val_array)
    return tf.Variable(initial, name=name)
