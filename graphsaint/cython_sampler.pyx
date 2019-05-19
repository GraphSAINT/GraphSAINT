# cython: language_level=3
# distutils: language=c++
# distutils: extra_compile_args = -fopenmp -std=c++11
# distutils: extra_link_args = -fopenmp
cimport cython
from cython.parallel import prange,parallel
from cython.operator import dereference as deref, preincrement as inc
from cython cimport Py_buffer
from libcpp.vector cimport vector
from libcpp.algorithm cimport sort, unique, lower_bound
from libcpp.map cimport map
from libcpp.unordered_map cimport unordered_map
from libcpp.utility cimport pair
import numpy as np
cimport numpy as np
from libc.stdio cimport printf
from libcpp cimport bool
import time,math
import random
from libc.stdlib cimport rand

cimport graphsaint.cython_utils as cutils
import graphsaint.cython_utils as cutils




cdef void _sampler_frontier(vector[int]& adj_indptr, vector[int]& adj_indices, \
        vector[int]& arr_deg, vector[int]& node_train,vector[vector[int]]& node_sampled, \
        int size_frontier, int size_subg, int avg_deg, int p, int num_rep) nogil:
    cdef vector[int] frontier
    cdef int i = 0
    cdef int num_train_node = node_train.size()
    cdef int r = 0
    cdef int alpha = 2
    cdef vector[int] arr_ind0
    cdef vector[int] arr_ind1
    cdef vector[int].iterator it
    arr_ind0.reserve(alpha*avg_deg)
    arr_ind1.reserve(alpha*avg_deg)
    cdef int c, cnt, j, k
    cdef int v, vidx, vpop, vneigh, offset, vnext
    cdef int idx_begin, idx_end
    cdef int num_neighs_pop, num_neighs_next
    while r < num_rep:
        # prepare initial frontier
        arr_ind0.clear()
        arr_ind1.clear()
        frontier.clear()
        i = 0
        while i < size_frontier:        # NB: here we don't care if a node appear twice
            frontier.push_back(node_train[rand()%num_train_node])
            i = i + 1
        # init indicator array
        it = frontier.begin()
        while it != frontier.end():
            v = deref(it)
            cnt = arr_ind0.size()
            c = cnt
            while c < cnt + arr_deg[v]:
                arr_ind0.push_back(v)
                arr_ind1.push_back(c-cnt)
                c = c + 1
            arr_ind1[cnt] = -arr_deg[v]
            inc(it)
        # iteratively update frontier
        j = size_frontier
        while j < size_subg:
            # select next node to pop out of frontier
            while True:
                vidx = rand()%arr_ind0.size()
                vpop = arr_ind0[vidx]
                if vpop >= 0:
                    break
            # prepare to update arr_ind*
            offset = arr_ind1[vidx]
            if offset < 0:
                idx_begin = vidx
                idx_end = idx_begin - offset
            else:
                idx_begin = vidx - offset
                idx_end = idx_begin - arr_ind1[idx_begin]
            # cleanup 1: invalidate entries
            k = idx_begin
            while k < idx_end:
                arr_ind0[k] = -1
                arr_ind1[k] = 0
                k = k + 1
            # cleanup 2: add new entries
            num_neighs_pop = adj_indptr[vpop+1] - adj_indptr[vpop]
            vnext = adj_indices[adj_indptr[vpop]+rand()%num_neighs_pop]
            node_sampled[p*num_rep+r].push_back(vnext)
            num_neighs_next = arr_deg[vnext]
            cnt = arr_ind0.size()
            c = cnt
            while c < cnt + num_neighs_next:
                arr_ind0.push_back(vnext)
                arr_ind1.push_back(c-cnt)
                c = c + 1
            arr_ind1[cnt] = -num_neighs_next
            j = j + 1
        node_sampled[p*num_rep+r].insert(node_sampled[p*num_rep+r].end(),frontier.begin(),frontier.end())
        sort(node_sampled[p*num_rep+r].begin(),node_sampled[p*num_rep+r].end())
        node_sampled[p*num_rep+r].erase(unique(node_sampled[p*num_rep+r].begin(),node_sampled[p*num_rep+r].end()),node_sampled[p*num_rep+r].end())
        r = r + 1




@cython.boundscheck(False)
@cython.wraparound(False)
def sampler_frontier_cython(np.ndarray[int,ndim=1,mode='c'] adj_indptr, np.ndarray[int,ndim=1,mode='c'] adj_indices,\
        np.ndarray[int,ndim=1,mode='c'] p_dist, np.ndarray[int,ndim=1,mode='c'] node_train,\
        int max_deg, int size_frontier, int size_subg, int num_proc, int num_sample_per_proc):
    """
    Done modifying this version --- keep doing for the others.
    """
    # prepare: common to all samplers
    cdef vector[int] adj_indptr_vec
    cutils.npy2vec_int(adj_indptr, adj_indptr_vec)
    cdef vector[int] adj_indices_vec
    cutils.npy2vec_int(adj_indices, adj_indices_vec)
    cdef vector[int] p_dist_vec
    cutils.npy2vec_int(p_dist, p_dist_vec)
    cdef vector[int] node_train_vec
    cutils.npy2vec_int(node_train, node_train_vec)
    #frontier = random.choices(node_train,k=size_frontier)
    #cdef vector[int] frontier_vec
    arr_deg = np.clip(p_dist,0,max_deg)
    cdef vector[int] arr_deg_vec
    cutils.npy2vec_int(arr_deg, arr_deg_vec)
    cdef int avg_deg = arr_deg.mean()
    cdef int p=0
    cdef vector[vector[int]] node_sampled
    node_sampled = vector[vector[int]](num_proc*num_sample_per_proc)
    cdef vector[vector[int]] ret_indptr
    ret_indptr = vector[vector[int]](num_proc*num_sample_per_proc)
    cdef vector[vector[int]] ret_indices
    ret_indices = vector[vector[int]](num_proc*num_sample_per_proc)
    # ch1
    cdef vector[vector[int]] ret_indices_orig
    ret_indices_orig = vector[vector[int]](num_proc*num_sample_per_proc)
    cdef vector[vector[float]] ret_data
    ret_data = vector[vector[float]](num_proc*num_sample_per_proc)

    with nogil, parallel(num_threads=num_proc):
        for p in prange(num_proc,schedule='dynamic'):
            _sampler_frontier(adj_indptr_vec,adj_indices_vec,arr_deg_vec,\
                node_train_vec,node_sampled,size_frontier,size_subg,avg_deg,p,num_sample_per_proc)
            # ch2
            cutils._adj_extract_cython(adj_indptr_vec,adj_indices_vec,node_sampled,ret_indptr,ret_indices,ret_indices_orig,ret_data,p,num_sample_per_proc)
    # prepare return values
    l_subg_indptr = list()
    l_subg_indices = list()
    # ch3
    l_subg_indices_orig = list()
    l_subg_data = list()
    l_subg_nodes = list()
    offset_nodes = [0]
    offset_indptr = [0]
    offset_indices = [0]
    offset_data = [0]
    for r in range(num_proc*num_sample_per_proc):
        offset_nodes.append(offset_nodes[r]+node_sampled[r].size())
        offset_indptr.append(offset_indptr[r]+ret_indptr[r].size())
        offset_indices.append(offset_indices[r]+ret_indices[r].size())
        offset_data.append(offset_data[r]+ret_data[r].size())
    cdef vector[int] ret_nodes_vec = vector[int]()
    cdef vector[int] ret_indptr_vec = vector[int]()
    cdef vector[int] ret_indices_vec = vector[int]()
    # ch4
    cdef vector[int] ret_indices_orig_vec = vector[int]()
    cdef vector[float] ret_data_vec = vector[float]()
    ret_nodes_vec.reserve(offset_nodes[num_proc*num_sample_per_proc])
    ret_indptr_vec.reserve(offset_indptr[num_proc*num_sample_per_proc])
    ret_indices_vec.reserve(offset_indices[num_proc*num_sample_per_proc])
    # ch5
    ret_indices_orig_vec.reserve(offset_indices[num_proc*num_sample_per_proc])
    ret_data_vec.reserve(offset_data[num_proc*num_sample_per_proc])
    for r in range(num_proc*num_sample_per_proc):
        ret_nodes_vec.insert(ret_nodes_vec.end(),node_sampled[r].begin(),node_sampled[r].end())
        ret_indptr_vec.insert(ret_indptr_vec.end(),ret_indptr[r].begin(),ret_indptr[r].end())
        ret_indices_vec.insert(ret_indices_vec.end(),ret_indices[r].begin(),ret_indices[r].end())
        # ch6
        ret_indices_orig_vec.insert(ret_indices_orig_vec.end(),ret_indices_orig[r].begin(),ret_indices_orig[r].end())
        ret_data_vec.insert(ret_data_vec.end(),ret_data[r].begin(),ret_data[r].end())

    cdef cutils.array_wrapper_int wint_indptr = cutils.array_wrapper_int()
    cdef cutils.array_wrapper_int wint_indices = cutils.array_wrapper_int()
    # ch7
    cdef cutils.array_wrapper_int wint_indices_orig = cutils.array_wrapper_int()
    cdef cutils.array_wrapper_int wint_nodes = cutils.array_wrapper_int()
    cdef cutils.array_wrapper_float wfloat_data = cutils.array_wrapper_float()

    wint_indptr.set_data(ret_indptr_vec)
    ret_indptr_np = np.frombuffer(wint_indptr,dtype=np.int32)
    wint_indices.set_data(ret_indices_vec)
    ret_indices_np = np.frombuffer(wint_indices,dtype=np.int32)
    # ch8
    wint_indices_orig.set_data(ret_indices_orig_vec)
    ret_indices_orig_np = np.frombuffer(wint_indices_orig,dtype=np.int32)
    wint_nodes.set_data(ret_nodes_vec)
    ret_nodes_np = np.frombuffer(wint_nodes,dtype=np.int32)
    wfloat_data.set_data(ret_data_vec)
    ret_data_np = np.frombuffer(wfloat_data,dtype=np.float32)

    for r in range(num_proc*num_sample_per_proc):
        l_subg_nodes.append(ret_nodes_np[offset_nodes[r]:offset_nodes[r+1]])
        l_subg_indptr.append(ret_indptr_np[offset_indptr[r]:offset_indptr[r+1]])
        l_subg_indices.append(ret_indices_np[offset_indices[r]:offset_indices[r+1]])
        # ch9
        l_subg_indices_orig.append(ret_indices_orig_np[offset_indices[r]:offset_indices[r+1]])
        l_subg_data.append(ret_data_np[offset_data[r]:offset_data[r+1]])
    # ch10
    return l_subg_indptr,l_subg_indices,l_subg_indices_orig,l_subg_data,l_subg_nodes





cdef void _sampler_khop(vector[int]& p_dist_cumsum,vector[int]& node_train,vector[vector[int]]& node_sampled, \
        int size_subg, int p, int num_rep) nogil:
    cdef int i = 0
    cdef int r = 0
    cdef int idx_subg
    cdef int sample
    cdef int rand_range = p_dist_cumsum[node_train.size()-1]

    while r < num_rep:
        idx_subg = p*num_rep+r
        i = 0
        while i < size_subg:
            sample = rand()%rand_range
            node_sampled[idx_subg].push_back(node_train[lower_bound(p_dist_cumsum.begin(),p_dist_cumsum.end(),sample)-p_dist_cumsum.begin()])
            i = i + 1
        r = r + 1
        sort(node_sampled[idx_subg].begin(),node_sampled[idx_subg].end())
        node_sampled[idx_subg].erase(unique(node_sampled[idx_subg].begin(),node_sampled[idx_subg].end()),node_sampled[idx_subg].end())


@cython.boundscheck(False)
@cython.wraparound(False)
def sampler_khop_cython(np.ndarray[int,ndim=1,mode='c'] adj_indptr, \
                        np.ndarray[int,ndim=1,mode='c'] adj_indices,\
                        np.ndarray[int,ndim=1,mode='c'] p_dist_cumsum, \
                        np.ndarray[int,ndim=1,mode='c'] node_train,\
                        int size_subg, int num_proc, int num_sample_per_proc):
    # ==== prepare start: common to all samplers ====
    cdef vector[int] adj_indptr_vec
    cutils.npy2vec_int(adj_indptr, adj_indptr_vec)
    cdef vector[int] adj_indices_vec
    cutils.npy2vec_int(adj_indices, adj_indices_vec)
    cdef vector[int] p_dist_cumsum_vec
    cutils.npy2vec_int(p_dist_cumsum, p_dist_cumsum_vec)
    cdef vector[int] node_train_vec
    cutils.npy2vec_int(node_train, node_train_vec)
    cdef int p=0
    cdef vector[vector[int]] node_sampled
    node_sampled = vector[vector[int]](num_proc*num_sample_per_proc)
    cdef vector[vector[int]] ret_indptr
    ret_indptr = vector[vector[int]](num_proc*num_sample_per_proc)
    cdef vector[vector[int]] ret_indices
    ret_indices = vector[vector[int]](num_proc*num_sample_per_proc)
    cdef vector[vector[int]] ret_indices_orig
    ret_indices_orig = vector[vector[int]](num_proc*num_sample_per_proc)
    cdef vector[vector[float]] ret_data
    ret_data = vector[vector[float]](num_proc*num_sample_per_proc)
    # ---- prepare end: common to all samplers ----
    print('start para khop sample')
    with nogil, parallel(num_threads=num_proc):
        for p in prange(num_proc,schedule='dynamic'):
            # **** the only thing you need to change ****
            _sampler_khop(p_dist_cumsum_vec,node_train_vec,node_sampled,size_subg,p,num_sample_per_proc)
            # *******************************************
            cutils._adj_extract_cython(adj_indptr_vec,adj_indices_vec,node_sampled,ret_indptr,ret_indices,ret_indices_orig,ret_data,p,num_sample_per_proc)
    print('end para khop sample')
    # ==== return start: common to all samplers ====
    l_subg_indptr = list()
    l_subg_indices = list()
    l_subg_indices_orig = list()
    l_subg_data = list()
    l_subg_nodes = list()
    offset_nodes = [0]
    offset_indptr = [0]
    offset_indices = [0]
    offset_data = [0]
    for r in range(num_proc*num_sample_per_proc):
        offset_nodes.append(offset_nodes[r]+node_sampled[r].size())
        offset_indptr.append(offset_indptr[r]+ret_indptr[r].size())
        offset_indices.append(offset_indices[r]+ret_indices[r].size())
        offset_data.append(offset_data[r]+ret_data[r].size())
    cdef vector[int] ret_nodes_vec = vector[int]()
    cdef vector[int] ret_indptr_vec = vector[int]()
    cdef vector[int] ret_indices_orig_vec = vector[int]()
    cdef vector[int] ret_indices_vec = vector[int]()
    cdef vector[float] ret_data_vec = vector[float]()
    ret_nodes_vec.reserve(offset_nodes[num_proc*num_sample_per_proc])
    ret_indptr_vec.reserve(offset_indptr[num_proc*num_sample_per_proc])
    ret_indices_vec.reserve(offset_indices[num_proc*num_sample_per_proc])
    ret_indices_orig_vec.reserve(offset_indices[num_proc*num_sample_per_proc])
    ret_data_vec.reserve(offset_data[num_proc*num_sample_per_proc])
    for r in range(num_proc*num_sample_per_proc):
        ret_nodes_vec.insert(ret_nodes_vec.end(),node_sampled[r].begin(),node_sampled[r].end())
        ret_indptr_vec.insert(ret_indptr_vec.end(),ret_indptr[r].begin(),ret_indptr[r].end())
        ret_indices_vec.insert(ret_indices_vec.end(),ret_indices[r].begin(),ret_indices[r].end())
        ret_indices_orig_vec.insert(ret_indices_orig_vec.end(),ret_indices_orig[r].begin(),ret_indices_orig[r].end())
        ret_data_vec.insert(ret_data_vec.end(),ret_data[r].begin(),ret_data[r].end())

    cdef cutils.array_wrapper_int wint_indptr = cutils.array_wrapper_int()
    cdef cutils.array_wrapper_int wint_indices = cutils.array_wrapper_int()
    cdef cutils.array_wrapper_int wint_indices_orig = cutils.array_wrapper_int()
    cdef cutils.array_wrapper_int wint_nodes = cutils.array_wrapper_int()
    cdef cutils.array_wrapper_float wfloat_data = cutils.array_wrapper_float()

    wint_indptr.set_data(ret_indptr_vec)
    ret_indptr_np = np.frombuffer(wint_indptr,dtype=np.int32)
    wint_indices.set_data(ret_indices_vec)
    ret_indices_np = np.frombuffer(wint_indices,dtype=np.int32)
    # ch8
    wint_indices_orig.set_data(ret_indices_orig_vec)
    ret_indices_orig_np = np.frombuffer(wint_indices_orig,dtype=np.int32)
    wint_nodes.set_data(ret_nodes_vec)
    ret_nodes_np = np.frombuffer(wint_nodes,dtype=np.int32)
    wfloat_data.set_data(ret_data_vec)
    ret_data_np = np.frombuffer(wfloat_data,dtype=np.float32)

    for r in range(num_proc*num_sample_per_proc):
        l_subg_nodes.append(ret_nodes_np[offset_nodes[r]:offset_nodes[r+1]])
        l_subg_indptr.append(ret_indptr_np[offset_indptr[r]:offset_indptr[r+1]])
        l_subg_indices.append(ret_indices_np[offset_indices[r]:offset_indices[r+1]])
        l_subg_indices_orig.append(ret_indices_orig_np[offset_indices[r]:offset_indices[r+1]])
        l_subg_data.append(ret_data_np[offset_data[r]:offset_data[r+1]])
    return l_subg_indptr,l_subg_indices,l_subg_indices_orig,l_subg_data,l_subg_nodes
    # ---- return end: common to all samplers ----










cdef void _sampler_rw(vector[int]& adj_indptr,\
                      vector[int]& adj_indices,\
                      vector[int]& node_train,\
                      vector[vector[int]]& node_sampled, \
                      int size_root, int size_depth, bool is_induced, int p, int num_rep) nogil:
    # TODO: if no graph induction step, then just return unsorted node
    cdef int iroot = 0
    cdef int idepth = 0
    cdef int r = 0
    cdef int idx_subg
    cdef int v
    cdef int num_train_node = node_train.size()
    while r < num_rep:
        idx_subg = p*num_rep+r
        # sample root
        iroot = 0
        while iroot < size_root:
            v = node_train[rand()%num_train_node]
            node_sampled[idx_subg].push_back(v)
            # sample random walk
            idepth = 0
            while idepth < size_depth:
                if (adj_indptr[v+1]-adj_indptr[v]>0):
                    v = adj_indices[adj_indptr[v]+rand()%(adj_indptr[v+1]-adj_indptr[v])]# neigh
                    node_sampled[idx_subg].push_back(v)
                elif not is_induced:
                    node_sampled[idx_subg].push_back(v)
                #   add self
                idepth = idepth + 1
            iroot = iroot + 1
        r = r + 1
        if is_induced:
            sort(node_sampled[idx_subg].begin(),node_sampled[idx_subg].end())
            node_sampled[idx_subg].erase(unique(node_sampled[idx_subg].begin(),node_sampled[idx_subg].end()),node_sampled[idx_subg].end())


@cython.boundscheck(False)
@cython.wraparound(False)
def sampler_rw_cython(np.ndarray[int,ndim=1,mode='c'] adj_indptr, \
                      np.ndarray[int,ndim=1,mode='c'] adj_indices,\
                      np.ndarray[int,ndim=1,mode='c'] node_train,\
                      int size_root, int size_depth, bool is_induced, int num_proc, int num_sample_per_proc):
    # ==== prepare start: common to all samplers ====
    cdef vector[int] adj_indptr_vec
    cutils.npy2vec_int(adj_indptr, adj_indptr_vec)
    cdef vector[int] adj_indices_vec
    cutils.npy2vec_int(adj_indices, adj_indices_vec)
    cdef vector[int] node_train_vec
    cutils.npy2vec_int(node_train, node_train_vec)
    cdef int p=0
    cdef vector[vector[int]] node_sampled
    node_sampled = vector[vector[int]](num_proc*num_sample_per_proc)
    cdef vector[vector[int]] ret_indptr
    ret_indptr = vector[vector[int]](num_proc*num_sample_per_proc)
    cdef vector[vector[int]] ret_indices
    ret_indices = vector[vector[int]](num_proc*num_sample_per_proc)
    # for non-induced version
    #cdef vector[vector[int]] ret_row
    #ret_row = vector[vector[int]](num_proc*num_sample_per_proc)
    #cdef vector[vector[int]] ret_col
    #ret_col = vector[vector[int]](num_proc*num_sample_per_proc)
    # ch1
    cdef vector[vector[int]] ret_indices_orig
    ret_indices_orig = vector[vector[int]](num_proc*num_sample_per_proc)
    cdef vector[vector[float]] ret_data
    ret_data = vector[vector[float]](num_proc*num_sample_per_proc)
    # ---- prepare end: common to all samplers ----
    with nogil, parallel(num_threads=num_proc):
        for p in prange(num_proc,schedule='dynamic'):
            # **** the only thing you need to change ****
            _sampler_rw(adj_indptr_vec,adj_indices_vec,node_train_vec,node_sampled,size_root,size_depth,is_induced,p,num_sample_per_proc)
            # *******************************************
            # ch2
            # TODO: should have a simpler extract method
            if not is_induced:
                cutils._adj_extract_ind_cython(adj_indptr_vec,node_sampled,ret_indptr,ret_indices,ret_indices_orig,ret_data,size_depth,p,num_sample_per_proc)
            else:
                cutils._adj_extract_cython(adj_indptr_vec,adj_indices_vec,node_sampled,ret_indptr,ret_indices,ret_indices_orig,ret_data,p,num_sample_per_proc)
    # ==== return start: common to all samplers ====
    l_subg_indptr = list()
    l_subg_indices = list()
    # ch3
    l_subg_indices_orig = list()
    l_subg_data = list()
    l_subg_nodes = list()
    offset_nodes = [0]
    offset_indptr = [0]
    offset_indices = [0]
    offset_data = [0]
    for r in range(num_proc*num_sample_per_proc):
        offset_nodes.append(offset_nodes[r]+node_sampled[r].size())
        offset_indptr.append(offset_indptr[r]+ret_indptr[r].size())
        offset_indices.append(offset_indices[r]+ret_indices[r].size())
        offset_data.append(offset_data[r]+ret_data[r].size())
    cdef vector[int] ret_nodes_vec = vector[int]()
    cdef vector[int] ret_indptr_vec = vector[int]()
    cdef vector[int] ret_indices_vec = vector[int]()
    # ch4
    cdef vector[int] ret_indices_orig_vec = vector[int]()
    cdef vector[float] ret_data_vec = vector[float]()
    ret_nodes_vec.reserve(offset_nodes[num_proc*num_sample_per_proc])
    ret_indptr_vec.reserve(offset_indptr[num_proc*num_sample_per_proc])
    ret_indices_vec.reserve(offset_indices[num_proc*num_sample_per_proc])
    # ch5
    ret_indices_orig_vec.reserve(offset_indices[num_proc*num_sample_per_proc])
    ret_data_vec.reserve(offset_data[num_proc*num_sample_per_proc])
    for r in range(num_proc*num_sample_per_proc):
        ret_nodes_vec.insert(ret_nodes_vec.end(),node_sampled[r].begin(),node_sampled[r].end())
        ret_indptr_vec.insert(ret_indptr_vec.end(),ret_indptr[r].begin(),ret_indptr[r].end())
        ret_indices_vec.insert(ret_indices_vec.end(),ret_indices[r].begin(),ret_indices[r].end())
        # ch6
        ret_indices_orig_vec.insert(ret_indices_orig_vec.end(),ret_indices_orig[r].begin(),ret_indices_orig[r].end())
        ret_data_vec.insert(ret_data_vec.end(),ret_data[r].begin(),ret_data[r].end())

    cdef cutils.array_wrapper_int wint_indptr = cutils.array_wrapper_int()
    cdef cutils.array_wrapper_int wint_indices = cutils.array_wrapper_int()
    # ch7
    cdef cutils.array_wrapper_int wint_indices_orig = cutils.array_wrapper_int()
    cdef cutils.array_wrapper_int wint_nodes = cutils.array_wrapper_int()
    cdef cutils.array_wrapper_float wfloat_data = cutils.array_wrapper_float()

    wint_indptr.set_data(ret_indptr_vec)
    ret_indptr_np = np.frombuffer(wint_indptr,dtype=np.int32)
    wint_indices.set_data(ret_indices_vec)
    ret_indices_np = np.frombuffer(wint_indices,dtype=np.int32)
    # ch8
    wint_indices_orig.set_data(ret_indices_orig_vec)
    ret_indices_orig_np = np.frombuffer(wint_indices_orig,dtype=np.int32)
    wint_nodes.set_data(ret_nodes_vec)
    ret_nodes_np = np.frombuffer(wint_nodes,dtype=np.int32)
    wfloat_data.set_data(ret_data_vec)
    ret_data_np = np.frombuffer(wfloat_data,dtype=np.float32)

    for r in range(num_proc*num_sample_per_proc):
        l_subg_nodes.append(ret_nodes_np[offset_nodes[r]:offset_nodes[r+1]])
        l_subg_indptr.append(ret_indptr_np[offset_indptr[r]:offset_indptr[r+1]])
        l_subg_indices.append(ret_indices_np[offset_indices[r]:offset_indices[r+1]])
        # ch9
        l_subg_indices_orig.append(ret_indices_orig_np[offset_indices[r]:offset_indices[r+1]])
        l_subg_data.append(ret_data_np[offset_data[r]:offset_data[r+1]])
    # ch10
    return l_subg_indptr,l_subg_indices,l_subg_indices_orig,l_subg_data,l_subg_nodes
    # ---- return end: common to all samplers ----




cdef void _sampler_edge(vector[int]& adj_indptr,\
                      vector[int]& adj_indices,\
                      vector[int]& node_train,\
                      vector[vector[int]]& node_sampled, \
                      int subg_size, vector[int]& indices_lut, int p, int num_rep) nogil:
    cdef int idx_subg
    cdef int r=0
    cdef int num_train_node=node_train.size()
    cdef int inode=0
    cdef int i=0
    while r<num_rep:
        idx_subg = p*num_rep+r
        i=0
        while i*2<subg_size:
            v=node_train[rand()%num_train_node]
            node_sampled[idx_subg].push_back(v)
            node_sampled[idx_subg].push_back(indices_lut[i])
            i=i+1
        r=r+1
        sort(node_sampled[idx_subg].begin(),node_sampled[idx_subg].end())
        node_sampled[idx_subg].erase(unique(node_sampled[idx_subg].begin(),node_sampled[idx_subg].end()),node_sampled[idx_subg].end())


@cython.boundscheck(False)
@cython.wraparound(False)
def sampler_edge_cython(np.ndarray[int,ndim=1,mode='c'] adj_indptr, \
                      np.ndarray[int,ndim=1,mode='c'] adj_indices,\
                      np.ndarray[int,ndim=1,mode='c'] node_train,\
                      int subg_size, np.ndarray[int,ndim=1,mode='c'] indices_lut,\
                      int num_proc, int num_sample_per_proc):
    # ==== prepare start: common to all samplers ====
    cdef vector[int] adj_indptr_vec
    cutils.npy2vec_int(adj_indptr, adj_indptr_vec)
    cdef vector[int] adj_indices_vec
    cutils.npy2vec_int(adj_indices, adj_indices_vec)
    cdef vector[int] node_train_vec
    cutils.npy2vec_int(node_train, node_train_vec)
    cdef vector[int] indices_lut_vec
    cutils.npy2vec_int(indices_lut, indices_lut_vec)
    cdef int p=0
    cdef vector[vector[int]] node_sampled
    node_sampled = vector[vector[int]](num_proc*num_sample_per_proc)
    cdef vector[vector[int]] ret_indptr
    ret_indptr = vector[vector[int]](num_proc*num_sample_per_proc)
    cdef vector[vector[int]] ret_indices
    ret_indices = vector[vector[int]](num_proc*num_sample_per_proc)
    # ch1
    cdef vector[vector[int]] ret_indices_orig
    ret_indices_orig = vector[vector[int]](num_proc*num_sample_per_proc)
    cdef vector[vector[float]] ret_data
    ret_data = vector[vector[float]](num_proc*num_sample_per_proc)
    # ---- prepare end: common to all samplers ----
    with nogil, parallel(num_threads=num_proc):
        for p in prange(num_proc,schedule='dynamic'):
            # **** the only thing you need to change ****
            _sampler_edge(adj_indptr_vec,adj_indices_vec,node_train_vec,node_sampled,subg_size,indices_lut_vec,p,num_sample_per_proc)
            # *******************************************
            # ch2
            cutils._adj_extract_cython(adj_indptr_vec,adj_indices_vec,node_sampled,ret_indptr,ret_indices,ret_indices_orig,ret_data,p,num_sample_per_proc)
    # ==== return start: common to all samplers ====
    l_subg_indptr = list()
    l_subg_indices = list()
    # ch3
    l_subg_indices_orig = list()
    l_subg_data = list()
    l_subg_nodes = list()
    offset_nodes = [0]
    offset_indptr = [0]
    offset_indices = [0]
    offset_data = [0]
    for r in range(num_proc*num_sample_per_proc):
        offset_nodes.append(offset_nodes[r]+node_sampled[r].size())
        offset_indptr.append(offset_indptr[r]+ret_indptr[r].size())
        offset_indices.append(offset_indices[r]+ret_indices[r].size())
        offset_data.append(offset_data[r]+ret_data[r].size())
    cdef vector[int] ret_nodes_vec = vector[int]()
    cdef vector[int] ret_indptr_vec = vector[int]()
    cdef vector[int] ret_indices_vec = vector[int]()
    # ch4
    cdef vector[int] ret_indices_orig_vec = vector[int]()
    cdef vector[float] ret_data_vec = vector[float]()
    ret_nodes_vec.reserve(offset_nodes[num_proc*num_sample_per_proc])
    ret_indptr_vec.reserve(offset_indptr[num_proc*num_sample_per_proc])
    ret_indices_vec.reserve(offset_indices[num_proc*num_sample_per_proc])
    # ch5
    ret_indices_orig_vec.reserve(offset_indices[num_proc*num_sample_per_proc])
    ret_data_vec.reserve(offset_data[num_proc*num_sample_per_proc])
    for r in range(num_proc*num_sample_per_proc):
        ret_nodes_vec.insert(ret_nodes_vec.end(),node_sampled[r].begin(),node_sampled[r].end())
        ret_indptr_vec.insert(ret_indptr_vec.end(),ret_indptr[r].begin(),ret_indptr[r].end())
        ret_indices_vec.insert(ret_indices_vec.end(),ret_indices[r].begin(),ret_indices[r].end())
        # ch6
        ret_indices_orig_vec.insert(ret_indices_orig_vec.end(),ret_indices_orig[r].begin(),ret_indices_orig[r].end())
        ret_data_vec.insert(ret_data_vec.end(),ret_data[r].begin(),ret_data[r].end())

    cdef cutils.array_wrapper_int wint_indptr = cutils.array_wrapper_int()
    cdef cutils.array_wrapper_int wint_indices = cutils.array_wrapper_int()
    # ch7
    cdef cutils.array_wrapper_int wint_indices_orig = cutils.array_wrapper_int()
    cdef cutils.array_wrapper_int wint_nodes = cutils.array_wrapper_int()
    cdef cutils.array_wrapper_float wfloat_data = cutils.array_wrapper_float()

    wint_indptr.set_data(ret_indptr_vec)
    ret_indptr_np = np.frombuffer(wint_indptr,dtype=np.int32)
    wint_indices.set_data(ret_indices_vec)
    ret_indices_np = np.frombuffer(wint_indices,dtype=np.int32)
    # ch8
    wint_indices_orig.set_data(ret_indices_orig_vec)
    ret_indices_orig_np = np.frombuffer(wint_indices_orig,dtype=np.int32)
    wint_nodes.set_data(ret_nodes_vec)
    ret_nodes_np = np.frombuffer(wint_nodes,dtype=np.int32)
    wfloat_data.set_data(ret_data_vec)
    ret_data_np = np.frombuffer(wfloat_data,dtype=np.float32)

    for r in range(num_proc*num_sample_per_proc):
        l_subg_nodes.append(ret_nodes_np[offset_nodes[r]:offset_nodes[r+1]])
        l_subg_indptr.append(ret_indptr_np[offset_indptr[r]:offset_indptr[r+1]])
        l_subg_indices.append(ret_indices_np[offset_indices[r]:offset_indices[r+1]])
        # ch9
        l_subg_indices_orig.append(ret_indices_orig_np[offset_indices[r]:offset_indices[r+1]])
        l_subg_data.append(ret_data_np[offset_data[r]:offset_data[r+1]])
    # ch10
    return l_subg_indptr,l_subg_indices,l_subg_indices_orig,l_subg_data,l_subg_nodes
    # ---- return end: common to all samplers ----


