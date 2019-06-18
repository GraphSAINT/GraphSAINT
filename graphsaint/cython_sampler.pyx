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



cdef class Sampler:
    cdef int num_proc,num_sample_per_proc
    cdef vector[int] adj_indptr_vec
    cdef vector[int] adj_indices_vec
    cdef vector[int] node_train_vec
    cdef vector[vector[int]] node_sampled
    cdef vector[vector[int]] ret_indptr
    cdef vector[vector[int]] ret_indices
    cdef vector[vector[int]] ret_indices_orig
    cdef vector[vector[float]] ret_data

    def __cinit__(self, np.ndarray[int,ndim=1,mode='c'] adj_indptr,
                        np.ndarray[int,ndim=1,mode='c'] adj_indices,
                        np.ndarray[int,ndim=1,mode='c'] node_train,
                        int num_proc, int num_sample_per_proc,*argv):
        cutils.npy2vec_int(adj_indptr,self.adj_indptr_vec)
        cutils.npy2vec_int(adj_indices,self.adj_indices_vec)
        cutils.npy2vec_int(node_train,self.node_train_vec)
        self.num_proc = num_proc
        self.num_sample_per_proc = num_sample_per_proc
        self.node_sampled = vector[vector[int]](num_proc*num_sample_per_proc)
        self.ret_indptr = vector[vector[int]](num_proc*num_sample_per_proc)
        self.ret_indices = vector[vector[int]](num_proc*num_sample_per_proc)
        self.ret_indices_orig = vector[vector[int]](num_proc*num_sample_per_proc)
        self.ret_data = vector[vector[float]](num_proc*num_sample_per_proc)

    cdef void adj_extract(self, int p) nogil:
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
        num_v_orig = self.adj_indptr_vec.size()-1
        while r < self.num_sample_per_proc:
            _arr_bit = vector[int](num_v_orig,-1)
            idx_g = p*self.num_sample_per_proc+r
            num_v_sub = self.node_sampled[idx_g].size()
            self.ret_indptr[idx_g] = vector[int](num_v_sub+1,0)
            self.ret_indices[idx_g] = vector[int]()
            self.ret_indices_orig[idx_g] = vector[int]()
            self.ret_data[idx_g] = vector[float]()
            i_end = num_v_sub
            i = 0
            while i < i_end:
                _arr_bit[self.node_sampled[idx_g][i]] = i
                i = i + 1
            i = 0
            while i < i_end:
                v = self.node_sampled[idx_g][i]
                start_neigh = self.adj_indptr_vec[v]
                end_neigh = self.adj_indptr_vec[v+1]
                j = start_neigh
                while j < end_neigh:
                    if _arr_bit[self.adj_indices_vec[j]] > -1:
                        self.ret_indices[idx_g].push_back(_arr_bit[self.adj_indices_vec[j]])
                        self.ret_indices_orig[idx_g].push_back(self.adj_indices_vec[j])
                        self.ret_indptr[idx_g][_arr_bit[v]+1] = self.ret_indptr[idx_g][_arr_bit[v]+1] + 1
                        self.ret_data[idx_g].push_back(1.)
                    j = j + 1
                i = i + 1
            cumsum = self.ret_indptr[idx_g][0]
            i = 0
            while i < i_end:
                cumsum = cumsum + self.ret_indptr[idx_g][i+1]
                self.ret_indptr[idx_g][i+1] = cumsum
                i = i + 1
            r = r + 1

    def get_return(self):
        # prepare return values
        num_subg = self.num_proc*self.num_sample_per_proc
        l_subg_indptr = list()
        l_subg_indices = list()
        l_subg_indices_orig = list()
        l_subg_data = list()
        l_subg_nodes = list()
        offset_nodes = [0]
        offset_indptr = [0]
        offset_indices = [0]
        offset_data = [0]
        for r in range(num_subg):
            offset_nodes.append(offset_nodes[r]+self.node_sampled[r].size())
            offset_indptr.append(offset_indptr[r]+self.ret_indptr[r].size())
            offset_indices.append(offset_indices[r]+self.ret_indices[r].size())
            offset_data.append(offset_data[r]+self.ret_data[r].size())
        cdef vector[int] ret_nodes_vec = vector[int]()
        cdef vector[int] ret_indptr_vec = vector[int]()
        cdef vector[int] ret_indices_vec = vector[int]()
        # ch4
        cdef vector[int] ret_indices_orig_vec = vector[int]()
        cdef vector[float] ret_data_vec = vector[float]()
        ret_nodes_vec.reserve(offset_nodes[num_subg])
        ret_indptr_vec.reserve(offset_indptr[num_subg])
        ret_indices_vec.reserve(offset_indices[num_subg])
        # ch5
        ret_indices_orig_vec.reserve(offset_indices[num_subg])
        ret_data_vec.reserve(offset_data[num_subg])
        for r in range(num_subg):
            ret_nodes_vec.insert(ret_nodes_vec.end(),self.node_sampled[r].begin(),self.node_sampled[r].end())
            ret_indptr_vec.insert(ret_indptr_vec.end(),self.ret_indptr[r].begin(),self.ret_indptr[r].end())
            ret_indices_vec.insert(ret_indices_vec.end(),self.ret_indices[r].begin(),self.ret_indices[r].end())
            # ch6
            ret_indices_orig_vec.insert(ret_indices_orig_vec.end(),self.ret_indices_orig[r].begin(),self.ret_indices_orig[r].end())
            ret_data_vec.insert(ret_data_vec.end(),self.ret_data[r].begin(),self.ret_data[r].end())

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

        for r in range(num_subg):
            l_subg_nodes.append(ret_nodes_np[offset_nodes[r]:offset_nodes[r+1]])
            l_subg_indptr.append(ret_indptr_np[offset_indptr[r]:offset_indptr[r+1]])
            l_subg_indices.append(ret_indices_np[offset_indices[r]:offset_indices[r+1]])
            # ch9
            l_subg_indices_orig.append(ret_indices_orig_np[offset_indices[r]:offset_indices[r+1]])
            l_subg_data.append(ret_data_np[offset_data[r]:offset_data[r+1]])
        # ch10
        return l_subg_indptr,l_subg_indices,l_subg_indices_orig,l_subg_data,l_subg_nodes

    cdef void sample(self, int p) nogil:
        pass

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def  par_sample(self):
        cdef int p = 0
        with nogil, parallel(num_threads=self.num_proc):
            for p in prange(self.num_proc,schedule='dynamic'):
                self.sample(p)
                self.adj_extract(p)
        return self.get_return()


# ----------------------------------------------------

cdef class MRW(Sampler):
    cdef int size_frontier,size_subg
    cdef int avg_deg
    cdef vector[int] arr_deg_vec
    def __cinit__(self, np.ndarray[int,ndim=1,mode='c'] adj_indptr,
                        np.ndarray[int,ndim=1,mode='c'] adj_indices,
                        np.ndarray[int,ndim=1,mode='c'] node_train,
                        int num_proc, int num_sample_per_proc,
                        np.ndarray[int,ndim=1,mode='c'] p_dist,
                        int max_deg, int size_frontier, int size_subg):
        self.size_frontier = size_frontier
        self.size_subg = size_subg
        _arr_deg = np.clip(p_dist,0,max_deg)
        cutils.npy2vec_int(_arr_deg,self.arr_deg_vec)
        self.avg_deg = _arr_deg.mean()

    cdef void sample(self, int p) nogil:
        cdef vector[int] frontier
        cdef int i = 0
        cdef int num_train_node = self.node_train_vec.size()
        cdef int r = 0
        cdef int alpha = 2
        cdef vector[int] arr_ind0
        cdef vector[int] arr_ind1
        cdef vector[int].iterator it
        arr_ind0.reserve(alpha*self.avg_deg)
        arr_ind1.reserve(alpha*self.avg_deg)
        cdef int c, cnt, j, k
        cdef int v, vidx, vpop, vneigh, offset, vnext
        cdef int idx_begin, idx_end
        cdef int num_neighs_pop, num_neighs_next
        while r < self.num_sample_per_proc:
            # prepare initial frontier
            arr_ind0.clear()
            arr_ind1.clear()
            frontier.clear()
            i = 0
            while i < self.size_frontier:        # NB: here we don't care if a node appear twice
                frontier.push_back(self.node_train_vec[rand()%num_train_node])
                i = i + 1
            # init indicator array
            it = frontier.begin()
            while it != frontier.end():
                v = deref(it)
                cnt = arr_ind0.size()
                c = cnt
                while c < cnt + self.arr_deg_vec[v]:
                    arr_ind0.push_back(v)
                    arr_ind1.push_back(c-cnt)
                    c = c + 1
                arr_ind1[cnt] = -self.arr_deg_vec[v]
                inc(it)
            # iteratively update frontier
            j = self.size_frontier
            while j < self.size_subg:
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
                num_neighs_pop = self.adj_indptr_vec[vpop+1] - self.adj_indptr_vec[vpop]
                vnext = self.adj_indices_vec[self.adj_indptr_vec[vpop]+rand()%num_neighs_pop]
                self.node_sampled[p*self.num_sample_per_proc+r].push_back(vnext)
                num_neighs_next = self.arr_deg_vec[vnext]
                cnt = arr_ind0.size()
                c = cnt
                while c < cnt + num_neighs_next:
                    arr_ind0.push_back(vnext)
                    arr_ind1.push_back(c-cnt)
                    c = c + 1
                arr_ind1[cnt] = -num_neighs_next
                j = j + 1
            self.node_sampled[p*self.num_sample_per_proc+r].insert(self.node_sampled[p*self.num_sample_per_proc+r].end(),frontier.begin(),frontier.end())
            sort(self.node_sampled[p*self.num_sample_per_proc+r].begin(),self.node_sampled[p*self.num_sample_per_proc+r].end())
            self.node_sampled[p*self.num_sample_per_proc+r].erase(unique(self.node_sampled[p*self.num_sample_per_proc+r].begin(),\
                    self.node_sampled[p*self.num_sample_per_proc+r].end()),self.node_sampled[p*self.num_sample_per_proc+r].end())
            r = r + 1



# ----------------------------------------------------

cdef class Node(Sampler):
    cdef int size_subg
    cdef vector[int] p_dist_cumsum_vec
    def __cinit__(self, np.ndarray[int,ndim=1,mode='c'] adj_indptr,
                        np.ndarray[int,ndim=1,mode='c'] adj_indices,
                        np.ndarray[int,ndim=1,mode='c'] node_train,
                        int num_proc, int num_sample_per_proc,
                        np.ndarray[int,ndim=1,mode='c'] p_dist_cumsum,
                        int size_subg):
        self.size_subg = size_subg
        cutils.npy2vec_int(p_dist_cumsum,self.p_dist_cumsum_vec)
    
    cdef void sample(self, int p) nogil:
        cdef int i = 0
        cdef int r = 0
        cdef int idx_subg
        cdef int sample
        cdef int rand_range = self.p_dist_cumsum_vec[self.node_train_vec.size()-1]
        while r < self.num_sample_per_proc:
            idx_subg = p*self.num_sample_per_proc+r
            i = 0
            while i < self.size_subg:
                sample = rand()%rand_range
                self.node_sampled[idx_subg].push_back(self.node_train_vec[lower_bound(self.p_dist_cumsum_vec.begin(),self.p_dist_cumsum_vec.end(),sample)-self.p_dist_cumsum_vec.begin()])
                i = i + 1
            r = r + 1
            sort(self.node_sampled[idx_subg].begin(),self.node_sampled[idx_subg].end())
            self.node_sampled[idx_subg].erase(unique(self.node_sampled[idx_subg].begin(),self.node_sampled[idx_subg].end()),self.node_sampled[idx_subg].end())

           


# ----------------------------------------------------

cdef class RW(Sampler):
    cdef int size_root, size_depth
    def __cinit__(self, np.ndarray[int,ndim=1,mode='c'] adj_indptr,
                        np.ndarray[int,ndim=1,mode='c'] adj_indices,
                        np.ndarray[int,ndim=1,mode='c'] node_train,
                        int num_proc, int num_sample_per_proc,
                        int size_root, int size_depth):
        self.size_root = size_root
        self.size_depth = size_depth
    
    cdef void sample(self, int p) nogil:
        cdef int iroot = 0
        cdef int idepth = 0
        cdef int r = 0
        cdef int idx_subg
        cdef int v
        cdef int num_train_node = self.node_train_vec.size()
        while r < self.num_sample_per_proc:
            idx_subg = p*self.num_sample_per_proc+r
            # sample root
            iroot = 0
            while iroot < self.size_root:
                v = self.node_train_vec[rand()%num_train_node]
                self.node_sampled[idx_subg].push_back(v)
                # sample random walk
                idepth = 0
                while idepth < self.size_depth:
                    if (self.adj_indptr_vec[v+1]-self.adj_indptr_vec[v]>0):
                        v = self.adj_indices_vec[self.adj_indptr_vec[v]+rand()%(self.adj_indptr_vec[v+1]-self.adj_indptr_vec[v])]
                        self.node_sampled[idx_subg].push_back(v)
                    #   add self
                    idepth = idepth + 1
                iroot = iroot + 1
            r = r + 1
            sort(self.node_sampled[idx_subg].begin(),self.node_sampled[idx_subg].end())
            self.node_sampled[idx_subg].erase(unique(self.node_sampled[idx_subg].begin(),self.node_sampled[idx_subg].end()),self.node_sampled[idx_subg].end())

           


# ----------------------------------------------------

cdef class Edge(Sampler):
    def __cinit__(self, np.ndarray[int,ndim=1,mode='c'] adj_indptr,
                        np.ndarray[int,ndim=1,mode='c'] adj_indices,
                        np.ndarray[int,ndim=1,mode='c'] node_train,
                        int num_proc, int num_sample_per_proc):
        pass








cdef void _sampler_khop(vector[int]& p_dist_cumsum,
                        vector[int]& node_train,
                        vector[vector[int]]& node_sampled,
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
def sampler_khop_cython(np.ndarray[int,ndim=1,mode='c'] adj_indptr,
                        np.ndarray[int,ndim=1,mode='c'] adj_indices,
                        np.ndarray[int,ndim=1,mode='c'] p_dist_cumsum,
                        np.ndarray[int,ndim=1,mode='c'] node_train,
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











cdef void _sampler_edge(vector[int]& adj_indptr,
                      vector[int]& adj_indices,
                      vector[int]& node_train,
                      vector[vector[int]]& node_sampled,
                      vector[int]& indices_lut, 
                      int subg_size, int p, int num_rep) nogil:
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
def sampler_edge_cython(np.ndarray[int,ndim=1,mode='c'] adj_indptr,
                        np.ndarray[int,ndim=1,mode='c'] adj_indices,
                        np.ndarray[int,ndim=1,mode='c'] node_train,
                        int subg_size, np.ndarray[int,ndim=1,mode='c'] indices_lut,
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
            _sampler_edge(adj_indptr_vec,adj_indices_vec,node_train_vec,node_sampled,indices_lut_vec,subg_size,p,num_sample_per_proc)
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


