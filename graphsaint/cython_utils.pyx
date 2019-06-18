# cython: language_level=3
# distutils: language=c++
# distutils: extra_compile_args = -fopenmp -std=c++11
# distutils: extra_link_args = -fopenmp
cimport cython
from cython.parallel import prange,parallel
from cython.operator import dereference, postincrement
from cython cimport Py_buffer
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.utility cimport pair
import numpy as np
cimport numpy as np
from libc.stdio cimport printf
import time

# reference: https://stackoverflow.com/questions/45133276/passing-c-vector-to-numpy-through-cython-without-copying-and-taking-care-of-me
cdef class array_wrapper_float:

    cdef void set_data(self, vector[float]& data):
        self.vec = move(data)

    # now implement the buffer protocol for the class
    # which makes it generally useful to anything that expects an array
    def __getbuffer__(self, Py_buffer *buffer, int flags):
        # relevant documentation http://cython.readthedocs.io/en/latest/src/userguide/buffer.html#a-matrix-class
        cdef Py_ssize_t itemsize = sizeof(self.vec[0])
        self.shape[0] = self.vec.size()
        self.strides[0] = sizeof(float)
        buffer.buf = <char *>&(self.vec[0])
        buffer.format = 'f'
        buffer.internal = NULL
        buffer.itemsize = itemsize
        buffer.len = self.vec.size() * itemsize
        buffer.ndim = 1
        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self.shape
        buffer.strides = self.strides
        buffer.suboffsets = NULL

    def __releasebuffer__(self,Py_buffer *buffer):
        pass


cdef class array_wrapper_int:

    cdef void set_data(self, vector[int]& data):
        self.vec = move(data)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        # relevant documentation http://cython.readthedocs.io/en/latest/src/userguide/buffer.html#a-matrix-class
        cdef Py_ssize_t itemsize = sizeof(self.vec[0])
        self.shape[0] = self.vec.size()
        self.strides[0] = sizeof(int)
        buffer.buf = <char *>&(self.vec[0])
        buffer.format = 'i'
        buffer.internal = NULL
        buffer.itemsize = itemsize
        buffer.len = self.vec.size() * itemsize
        buffer.ndim = 1
        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self.shape
        buffer.strides = self.strides
        buffer.suboffsets = NULL

    def __releasebuffer__(self,Py_buffer *buffer):
        pass


cdef void _adj_extract_cython(vector[int]& adj_indptr, vector[int]& adj_indices,vector[vector[int]]& node_sampled,\
        vector[vector[int]]& ret_indptr, vector[vector[int]]& ret_indices, vector[vector[int]]& ret_indices_orig,\
        vector[vector[float]]& ret_data, int p, int num_rep, vector[vector[int]]& ret_edge_index) nogil:
    """
    Extract a subg adj matrix from the original adj matrix
    ret_indices_orig:   the indices vector corresponding to node id in original G.
    """
    cdef int r = 0
    cdef int idx_g = 0
    cdef int i, i_end, v, j
    cdef int num_v_orig, num_v_sub
    cdef int start_neigh, end_neigh
    cdef vector[int] _arr_bit
    cdef int cumsum
    num_v_orig = adj_indptr.size()-1
    while r < num_rep:
        _arr_bit = vector[int](num_v_orig,-1)
        idx_g = p*num_rep+r
        num_v_sub = node_sampled[idx_g].size()
        ret_indptr[idx_g] = vector[int](num_v_sub+1,0)
        ret_indices[idx_g] = vector[int]()
        ret_indices_orig[idx_g] = vector[int]()
        ret_data[idx_g] = vector[float]()
        ret_edge_index[idx_g] = vector[int]()
        i_end = num_v_sub
        i = 0
        while i < i_end:
            _arr_bit[node_sampled[idx_g][i]] = i
            i = i + 1
        i = 0
        while i < i_end:
            v = node_sampled[idx_g][i]
            start_neigh = adj_indptr[v]
            end_neigh = adj_indptr[v+1]
            j = start_neigh
            while j < end_neigh:
                if _arr_bit[adj_indices[j]] > -1:
                    ret_indices[idx_g].push_back(_arr_bit[adj_indices[j]])
                    ret_indices_orig[idx_g].push_back(adj_indices[j])
                    ret_edge_index[idx_g].push_back(j)
                    ret_indptr[idx_g][_arr_bit[v]+1] = ret_indptr[idx_g][_arr_bit[v]+1] + 1
                    ret_data[idx_g].push_back(1.)
                j = j + 1
            i = i + 1
        cumsum = ret_indptr[idx_g][0]
        i = 0
        while i < i_end:
            cumsum = cumsum + ret_indptr[idx_g][i+1]
            ret_indptr[idx_g][i+1] = cumsum
            i = i + 1
        r = r + 1

# temp for test
cdef void _adj_extract_cython_old(vector[int]& adj_indptr, vector[int]& adj_indices,vector[vector[int]]& node_sampled,\
        vector[vector[int]]& ret_indptr, vector[vector[int]]& ret_indices, vector[vector[int]]& ret_indices_orig,\
        vector[vector[float]]& ret_data, int p, int num_rep) nogil:
    """
    Extract a subg adj matrix from the original adj matrix
    ret_indices_orig:   the indices vector corresponding to node id in original G.
    """
    cdef int r = 0
    cdef int idx_g = 0
    cdef int i, i_end, v, j
    cdef int num_v_orig, num_v_sub
    cdef int start_neigh, end_neigh
    cdef vector[int] _arr_bit
    cdef int cumsum
    num_v_orig = adj_indptr.size()-1
    while r < num_rep:
        _arr_bit = vector[int](num_v_orig,-1)
        idx_g = p*num_rep+r
        num_v_sub = node_sampled[idx_g].size()
        ret_indptr[idx_g] = vector[int](num_v_sub+1,0)
        ret_indices[idx_g] = vector[int]()
        ret_indices_orig[idx_g] = vector[int]()
        ret_data[idx_g] = vector[float]()
        i_end = num_v_sub
        i = 0
        while i < i_end:
            _arr_bit[node_sampled[idx_g][i]] = i
            i = i + 1
        i = 0
        while i < i_end:
            v = node_sampled[idx_g][i]
            start_neigh = adj_indptr[v]
            end_neigh = adj_indptr[v+1]
            j = start_neigh
            while j < end_neigh:
                if _arr_bit[adj_indices[j]] > -1:
                    ret_indices[idx_g].push_back(_arr_bit[adj_indices[j]])
                    ret_indices_orig[idx_g].push_back(adj_indices[j])
                    ret_indptr[idx_g][_arr_bit[v]+1] = ret_indptr[idx_g][_arr_bit[v]+1] + 1
                    ret_data[idx_g].push_back(1.)
                j = j + 1
            i = i + 1
        cumsum = ret_indptr[idx_g][0]
        i = 0
        while i < i_end:
            cumsum = cumsum + ret_indptr[idx_g][i+1]
            ret_indptr[idx_g][i+1] = cumsum
            i = i + 1
        r = r + 1


cdef void _adj_extract_ind_cython(vector[int]& adj_indptr,vector[vector[int]]& node_sampled,\
        vector[vector[int]]& ret_row, vector[vector[int]]& ret_col, vector[vector[int]]& ret_indices_orig, vector[vector[float]]& ret_data, \
        int depth_walk, int p, int num_rep) nogil:
    """
    THIS IS ONLY TO BE USED WITH RW SAMPLER. FOR NON-INDUCTION SAMPLING
    Extract a subg adj matrix from the original adj matrix
    ret_indices_orig:   the indices vector corresponding to node id in original G.
    """
    cdef int r = 0
    cdef int idx_g = 0
    cdef int i, i_end, v, j, ii
    cdef int size_dummy
    cdef int num_v_orig, num_v_sub
    cdef int start_neigh, end_neigh
    cdef vector[int] _arr_bit
    cdef int cumsum
    num_v_orig = adj_indptr.size()-1
    while r < num_rep:
        _arr_bit = vector[int](num_v_orig,-1)
        idx_g = p*num_rep+r
        num_v_sub = node_sampled[idx_g].size()
        ret_row[idx_g] = vector[int]()
        ret_col[idx_g] = vector[int]()
        ret_indices_orig[idx_g] = vector[int]()
        ret_data[idx_g] = vector[float]()
        i_end = num_v_sub
        i = 0
        ii = 0
        while i < i_end:
            if _arr_bit[node_sampled[idx_g][i]] == -1:
                _arr_bit[node_sampled[idx_g][i]] = ii        # setup remapper
                ii = ii + 1
            i = i + 1
        i = 0
        while i < i_end:
            ret_indices_orig[idx_g].push_back(_arr_bit[node_sampled[idx_g][i]])
            if i % (depth_walk+1) == depth_walk:
                i = i + 1
                continue
            ret_row[idx_g].push_back(_arr_bit[node_sampled[idx_g][i]])
            ret_col[idx_g].push_back(_arr_bit[node_sampled[idx_g][i+1]])
            ret_row[idx_g].push_back(_arr_bit[node_sampled[idx_g][i+1]])
            ret_col[idx_g].push_back(_arr_bit[node_sampled[idx_g][i]])
            ret_data[idx_g].push_back(1.)
            ret_data[idx_g].push_back(1.)
            i = i + 1
        # append dummy vals at the end of ret_indices_orig, to comply with API
        size_dummy = ret_col[idx_g].size()
        i = ret_indices_orig[idx_g].size()
        while i < size_dummy:
            ret_indices_orig[idx_g].push_back(-1)
            i = i + 1
        r = r + 1
        # for iterating node_sampled[idx_g]
        #   if % (depth_walk+1) == depth_walk:
        #       skip
        #   else:
        #       added to subg

