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
from libcpp.unordered_map cimport unordered_map
from libcpp.utility cimport pair
import numpy as np
cimport numpy as np
from libc.stdio cimport printf
from libcpp cimport bool
import time,math

cimport graphsaint.cython_utils as cutils
import graphsaint.cython_utils as cutils



cdef void _adj_norm(vector[int]& adj_indptr, vector[float]& adj_data, vector[float]& scale,
        int p, int stride, int num_v) nogil:
    cdef int _row_begin,_row_end
    cdef int idx_end = (p+1)*stride if (p+1)*stride<num_v else num_v
    cdef int row = p*stride
    cdef int col
    while row < idx_end:
        _row_begin = adj_indptr[row]
        _row_end = adj_indptr[row+1]
        if _row_begin == _row_end:
            row = row + 1
            continue
        col = _row_begin
        while col < _row_end:
            adj_data[col] = adj_data[col]*scale[row]
            col = col + 1
        row = row + 1

def adj_norm_cython(np.ndarray[int,ndim=1,mode="c"] adj_indptr, np.ndarray[int,ndim=1,mode="c"] adj_indices, \
        np.ndarray[float,ndim=1,mode="c"] adj_data, scale, num_proc=10):
    t00 = time.time()
    cdef vector[int] adj_indptr_vec
    cutils.npy2vec_int(adj_indptr,adj_indptr_vec)
    cdef vector[int] adj_indices_vec
    cutils.npy2vec_int(adj_indices,adj_indices_vec)
    cdef vector[float] adj_data_vec
    cutils.npy2vec_float(adj_data,adj_data_vec)
    cdef vector[float] scale_c = scale
    cdef int num_proc_c = num_proc
    cdef int p = 0
    cdef int num_v = len(adj_indptr)-1
    cdef int stride = (num_v/num_proc)+1
    t000 = time.time()
    print('prepare time: ',t000-t00)
    printf("num processors: %d, stride: %d\n",num_proc_c,stride)
    with nogil,parallel(num_threads=num_proc_c):
        for p in prange(num_proc_c,schedule='dynamic'):
            _adj_norm(adj_indptr_vec,adj_data_vec,scale_c,p,stride,num_v)
    t0 = time.time()
    cdef cutils.array_wrapper_float wfloat = cutils.array_wrapper_float()
    wfloat.set_data(adj_data_vec)
    adj_data_np = np.frombuffer(wfloat, dtype=np.float32)
    t1 = time.time()
    print("time to convert from vector to ndarray: ",t1-t0)
    return adj_data_np




@cython.boundscheck(False)
@cython.wraparound(False)
def adj_extract_cython(np.ndarray[int,ndim=1,mode="c"] adj_indptr, np.ndarray[int,ndim=1,mode="c"] adj_indices,\
        np.ndarray[int,ndim=1,mode="c"] node_subgraph, num_proc=10):
    cdef vector[int] adj_indptr_vec
    cutils.npy2vec_int(adj_indptr,adj_indptr_vec)
    cdef vector[int] adj_indices_vec
    cutils.npy2vec_int(adj_indices,adj_indices_vec)
    cdef vector[int] node_subgraph_vec
    cutils.npy2vec_int(node_subgraph,node_subgraph_vec)
    cdef int num_proc_c = num_proc
    # start prepare sub-adj
    cdef vector[int] sub_indptr # TODO: reserve??
    cutils.npy2vec_int(np.zeros(node_subgraph_vec.size()+1,dtype=np.int32),sub_indptr)
    cdef vector[int] sub_indices
    cdef vector[float] sub_data
    cdef int num_v_orig = adj_indptr_vec.size()-1
    cdef int num_v_sub = node_subgraph_vec.size()
    cdef vector[int] _arr_bit
    cutils.npy2vec_int(np.zeros(num_v_orig,dtype=np.int32)-1,_arr_bit)
    cdef int i = 0
    cdef int i_end
    cdef int idx, idx_end, v
    idx_end = num_v_sub
    idx = 0
    while idx < idx_end:
        _arr_bit[node_subgraph_vec[idx]] = idx
        idx = idx + 1
    idx_end = num_v_sub
    idx = 0
    while idx < idx_end:
        v = node_subgraph_vec[idx]
        i = adj_indptr_vec[v]
        i_end = adj_indptr_vec[v+1]
        while i < i_end:
            if _arr_bit[adj_indices[i]] > -1:
                sub_indices.push_back(_arr_bit[adj_indices[i]])
                sub_indptr[_arr_bit[v]+1] = sub_indptr[_arr_bit[v]+1] + 1
                sub_data.push_back(1.)
            i = i + 1
        idx = idx + 1
    cdef cutils.array_wrapper_int windptr = cutils.array_wrapper_int()
    windptr.set_data(sub_indptr)
    sub_indptr_np = np.frombuffer(windptr, dtype=np.int32).cumsum()
    cdef cutils.array_wrapper_int windices = cutils.array_wrapper_int()
    windices.set_data(sub_indices)
    sub_indices_np = np.frombuffer(windices, dtype=np.int32)
    cdef cutils.array_wrapper_float wdata = cutils.array_wrapper_float()
    wdata.set_data(sub_data)
    sub_data_np = np.frombuffer(wdata, dtype=np.float32)
    return sub_indptr_np, sub_indices_np, sub_data_np


"""
cdef void _adj_extract_cython(vector[int]& adj_indptr, vector[int]& adj_indices,vector[vector[int]]& node_sampled,\
        vector[vector[int]]& ret_indptr, vector[vector[int]]& ret_indices, vector[vector[float]]& ret_data, \
        int p, int num_rep) nogil:
    cdef int r = 0
    cdef int idx_g = 0
    cdef int i, i_end, v, j
    cdef int num_v_orig, num_v_sub
    cdef int start_neigh, end_neigh
    cdef vector[int] _arr_bit
    cdef int cumsum
    _arr_bit = vector[int](num_v_orig,-1)
    num_v_orig = adj_indptr.size()-1
    while r < num_rep:
        _arr_bit.clear()
        idx_g = p*num_rep+r
        num_v_sub = node_sampled[idx_g].size()
        ret_indptr[idx_g] = vector[int](num_v_sub+1,0)
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
                    ret_indptr[idx_g][_arr_bit[v]+1] = ret_indptr[idx_g][_arr_bit[v]+1] + 1
                    ret_data[idx_g].push_back(1.)
                j = j + 1
            i = i + 1
        cumsum = ret_indptr[idx_g][0]
        i = 0
        while i < i_end:
            cumsum = cumsum + ret_indptr[idx_g][i+1]
            ret_indptr[idx_g][i+1] = cumsum
        r = r + 1
    """
"""
    # --------- outdated below ---------------
    # start prepare sub-adj
    cdef vector[int] sub_indptr # TODO: reserve??
    cutils.npy2vec_int(np.zeros(node_subgraph_vec.size()+1,dtype=np.int32),sub_indptr)
    cdef vector[int] sub_indices
    cdef vector[float] sub_data
    cdef int num_v_orig = adj_indptr_vec.size()-1
    cdef int num_v_sub = node_subgraph_vec.size()
    cdef vector[int] _arr_bit
    _arr_bit = vector[int](num_v_orig,-1)
    cdef int i = 0
    cdef int i_end
    cdef int idx, idx_end, v
    idx_end = num_v_sub
    idx = 0
    while idx < idx_end:
        _arr_bit[node_subgraph_vec[idx]] = idx
        idx = idx + 1
    idx = 0
    while idx < idx_end:
        v = node_subgraph_vec[idx]
        i = adj_indptr_vec[v]
        i_end = adj_indptr_vec[v+1]
        while i < i_end:
            if _arr_bit[adj_indices[i]] > -1:
                sub_indices.push_back(_arr_bit[adj_indices[i]])
                sub_indptr[_arr_bit[v]+1] = sub_indptr[_arr_bit[v]+1] + 1
                sub_data.push_back(1.)
            i = i + 1
        idx = idx + 1
    cdef cutils.array_wrapper_int windptr = cutils.array_wrapper_int()
    windptr.set_data(sub_indptr)
    sub_indptr_np = np.frombuffer(windptr, dtype=np.int32).cumsum()
    cdef cutils.array_wrapper_int windices = cutils.array_wrapper_int()
    windices.set_data(sub_indices)
    sub_indices_np = np.frombuffer(windices, dtype=np.int32)
    cdef cutils.array_wrapper_float wdata = cutils.array_wrapper_float()
    wdata.set_data(sub_data)
    sub_data_np = np.frombuffer(wdata, dtype=np.float32)
    return sub_indptr_np, sub_indices_np, sub_data_np
    """
