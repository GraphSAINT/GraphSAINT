# cython: language_level=3
# distutils: language=c++
# distutils: extra_compile_args = -fopenmp -std=c++11
# distutils: extra_link_args = -fopenmp

from cython.parallel import prange,parallel
from cython.operator import dereference, postincrement
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.unordered_map cimport unordered_map
from libcpp.utility cimport pair
import numpy as np
cimport numpy as np
from libc.stdio cimport printf
from libcpp cimport bool


cdef void __adj_mult_main_1(vector[int]& adj1_indptr, vector[int]& adj1_indices, vector[int]& adj1_data,
            vector[int]& adj2_indptr, vector[int]& adj2_indices, vector[int]& adj2_data,
            vector[int]& ret_indptr, vector[vector[int]]& ret_indices, vector[vector[int]]& ret_data,
            int p, int stride, int num_v) nogil:
    cdef int _row_begin,_row_end,_col_begin,_col_end
    cdef int ptr1,ptr2,data
    cdef int row = p*stride
    cdef int col
    cdef int idx_end = (p+1)*stride if (p+1)*stride<num_v else num_v
    while row < idx_end:
        if (row%1000 == 0):
            printf("finish: %d\n",row)
        _row_begin = adj1_indptr[row]
        _row_end = adj1_indptr[row+1]
        if _row_begin == _row_end:
            row = row + 1
            continue
        col = 0
        while col < num_v:
            _col_begin = adj2_indptr[col]
            _col_end = adj2_indptr[col+1]
            if _col_begin == _col_end:
                col = col + 1
                continue
            if adj1_indices[_row_end-1]<adj2_indices[_col_begin] \
                or adj1_indices[_row_begin]>adj2_indices[_col_end-1]:
                col = col + 1
                continue
            # compute data
            ptr1 = _row_begin; ptr2 = _col_begin;
            data = 0
            while ptr1!=_row_end and ptr2!=_col_end:
                if adj1_indices[ptr1] > adj2_indices[ptr2]:
                    ptr2 = ptr2 + 1
                elif adj1_indices[ptr1] < adj2_indices[ptr2]:
                    ptr1 = ptr1 + 1
                else:
                    data = data + adj1_data[ptr1]*adj2_data[ptr2]
                    ptr1 = ptr1 + 1; ptr2 = ptr2 + 1;
            if data == 0:
                col = col + 1
                continue
            ret_indptr[row] = ret_indptr[row] + 1
            ret_indices[p].push_back(col)
            ret_data[p].push_back(data)
            col = col + 1
        row = row + 1


cdef void __adj_mult_main_2(vector[int]& adj_indptr, vector[int]& adj_indices, vector[int]& adj_data,
            vector[int]& ret_indptr, vector[vector[int]]& ret_indices, vector[vector[int]]& ret_data,
            int p, int stride, int num_v) nogil:
    cdef int row = p*stride
    cdef int idx_end = (p+1)*stride if (p+1)*stride<num_v else num_v
    cdef int idx_hop1_begin,idx_hop1_end,idx_hop2_begin,idx_hop2_end
    cdef int row_hop1,row_hop2
    cdef int v_hop1
    cdef unordered_map[int,int] count
    cdef unordered_map[int,int].iterator it
    cdef int _key
    cdef pair[int,int] _pair
    while row < idx_end:
        if (row%1000 == 0):
            printf("finish: %d\n",row)
        idx_hop1_begin = adj_indptr[row]
        idx_hop1_end = adj_indptr[row+1]
        row_hop1 = idx_hop1_begin
        count.clear()
        while row_hop1 < idx_hop1_end:
            v_hop1 = adj_indices[row_hop1]
            idx_hop2_begin = adj_indptr[v_hop1]
            idx_hop2_end = adj_indptr[v_hop1+1]
            # need tp append 2 hop neighbors
            row_hop2 = idx_hop2_begin
            while row_hop2 < idx_hop2_end:
                _key = adj_indices[row_hop2]
                if count.find(_key) == count.end():
                    _pair.first = _key
                    _pair.second = adj_data[row_hop1]*adj_data[row_hop2]#1
                    count.insert(_pair)
                else:
                    count[_key] = count[_key] + adj_data[row_hop1]*adj_data[row_hop2]#1
                row_hop2 = row_hop2 + 1
            row_hop1 = row_hop1 + 1
        # setup data structure
        ret_indptr[row] = count.size()
        it = count.begin()
        while (it != count.end()):
            ret_indices[p].push_back(dereference(it).first)
            ret_data[p].push_back(dereference(it).second)
            postincrement(it)
        row = row + 1


cdef void __adj_mult_main_2_simple(vector[int]& adj_indptr, vector[int]& adj_indices, vector[int]& adj_data,
            vector[int]& ret_indptr, vector[vector[int]]& ret_indices, vector[vector[int]]& ret_data,
            int p, int stride, int num_v) nogil:
    cdef int row = p*stride
    cdef int idx_end = (p+1)*stride if (p+1)*stride<num_v else num_v
    cdef int idx_hop1_begin,idx_hop1_end,idx_hop2_begin,idx_hop2_end
    cdef int row_hop1,row_hop2
    cdef int v_hop1
    cdef unordered_map[int,int] count
    cdef unordered_map[int,int].iterator it
    cdef int _key
    cdef pair[int,int] _pair
    while row < idx_end:
        if (row%1000 == 0):
            printf("finish: %d\n",row)
        idx_hop1_begin = adj_indptr[row]
        idx_hop1_end = adj_indptr[row+1]
        row_hop1 = idx_hop1_begin
        count.clear()
        while row_hop1 < idx_hop1_end:
            v_hop1 = adj_indices[row_hop1]
            idx_hop2_begin = adj_indptr[v_hop1]
            idx_hop2_end = adj_indptr[v_hop1+1]
            # need tp append 2 hop neighbors
            row_hop2 = idx_hop2_begin
            while row_hop2 < idx_hop2_end:
                _key = adj_indices[row_hop2]
                if count.find(_key) == count.end():
                    _pair.first = _key
                    _pair.second = 1
                    count.insert(_pair)
                else:
                    count[_key] = count[_key] + 1
                row_hop2 = row_hop2 + 1
            row_hop1 = row_hop1 + 1
        # setup data structure
        ret_indptr[row] = count.size()
        it = count.begin()
        while (it != count.end()):
            ret_indices[p].push_back(dereference(it).first)
            ret_data[p].push_back(dereference(it).second)
            postincrement(it)
        row = row + 1



def _adj_mult_method1(adj1,adj2,num_proc=10):
    cdef vector[int] adj1_indptr = adj1[0]
    cdef vector[int] adj1_indices = adj1[1]
    cdef vector[int] adj1_data = adj1[2]
    cdef vector[int] adj2_indptr = adj2[0]
    cdef vector[int] adj2_indices = adj2[1]
    cdef vector[int] adj2_data = adj2[2]
    assert len(adj1[0]) == len(adj2[0])
    cdef int num_v = len(adj1[0])-1
    cdef int stride = (num_v/num_proc)+1
    cdef vector[int] ret_indptr = [0]*num_v
    cdef vector[vector[int]] ret_indices = [[] for p in range(num_proc)]
    cdef vector[vector[int]] ret_data = [[] for p in range(num_proc)]
    cdef int p = 0
    cdef int num_proc_c = num_proc
    printf("# threads: %d\n",num_proc_c)
    with nogil,parallel(num_threads=num_proc_c):
        for p in prange(num_proc_c,schedule='dynamic'):
            __adj_mult_main_1(adj1_indptr,adj1_indices,adj1_data,\
                    adj2_indptr,adj2_indices,adj2_data,\
                    ret_indptr,ret_indices,ret_data,p,stride,num_v)
    l_ret_indptr = [v for v in ret_indptr]
    l_ret_indptr.insert(0,0)
    l_ret_indptr = list(np.array(l_ret_indptr).cumsum())
    l_ret_indices = []
    for p in range(num_proc):
        l_ret_indices.extend(ret_indices[p])
    l_ret_data = []
    for p in range(num_proc):
        l_ret_data.extend(ret_data[p])
    return (l_ret_indptr,l_ret_indices,l_ret_data)


def _adj_pow2(adj,num_proc=10,is_adj_val1=False):
    cdef vector[int] adj_indptr = adj[0]
    cdef vector[int] adj_indices = adj[1]
    cdef vector[int] adj_data = adj[2]
    cdef int num_v = len(adj[0])-1
    cdef int stride = (num_v/num_proc)+1
    cdef vector[int] ret_indptr = [0]*num_v
    cdef vector[vector[int]] ret_indices = [[] for p in range(num_proc)]
    cdef vector[vector[int]] ret_data = [[] for p in range(num_proc)]
    cdef int p = 0
    cdef int num_proc_c = num_proc  # since nogil requires no py obj
    printf("# threads: %d\n",num_proc_c)
    cdef bool is_adj_val1_c = is_adj_val1
    with nogil,parallel(num_threads=num_proc_c):
        for p in prange(num_proc_c,schedule='dynamic'):
            if is_adj_val1_c:
                __adj_mult_main_2_simple(adj_indptr,adj_indices,adj_data,\
                    ret_indptr,ret_indices,ret_data,p,stride,num_v)
            else:
                __adj_mult_main_2(adj_indptr,adj_indices,adj_data,\
                    ret_indptr,ret_indices,ret_data,p,stride,num_v)
    l_ret_indptr = [v for v in ret_indptr]
    l_ret_indptr.insert(0,0)
    l_ret_indptr = list(np.array(l_ret_indptr).cumsum())
    l_ret_indices = []
    for p in range(num_proc):
        l_ret_indices.extend(ret_indices[p])
    l_ret_data = []
    for p in range(num_proc):
        l_ret_data.extend(ret_data[p])
    return (l_ret_indptr,l_ret_indices,l_ret_data)


def get_adj_pow(adj,power=2,num_proc=10,method='tree',is_adj_val1=False):
    """
    Assume a symmetric adj now. 
    Also, remember to check if the adj in supervised_train has entry 1 or 1/N
    is_adj_val1: True if all values in adj are either 0 or 1
    """
    adj2 = (adj.indptr,adj.indices,adj.data)
    if power == 2 and method == 'tree':
        adj2 = _adj_pow2(adj2,num_proc=num_proc,is_adj_val1=is_adj_val1)
    else:
        for p in range(power-1):
            adj2 = _adj_mult_method1((adj.indptr,adj.indices,adj.data),adj2,num_proc=num_proc)
    return adj2
