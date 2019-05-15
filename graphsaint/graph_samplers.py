from graphsaint.globals import *
import numpy as np
import json
import random
from scipy.stats import rv_discrete
import scipy.sparse
from zython.logf.printf import printf
import abc
import time
import math
import pdb
from math import ceil
import concurrent.futures as cuf
import graphsaint.cython_sampler as cy

NUM_PROC = 20
RUN_PER_PROC = 10

class graph_sampler:
    __metaclass__ = abc.ABCMeta
    def __init__(self,adj_train,adj_full,node_train,size_subgraph,args_preproc):
        assert adj_train.shape == adj_full.shape
        self.adj_train = adj_train
        self.adj_full = adj_full
        self.node_train = np.unique(node_train).astype(np.int32)
        # size in terms of number of vertices in subgraph
        self.size_subgraph = size_subgraph
        self.coverage_v_train = np.array([float('inf')]*(self.adj_full.shape[0]))
        self.coverage_v_train[self.node_train] = 0.
        self.name_sampler = 'None'
        self.node_subgraph = None
        # =======================
        self.arr_deg_train = np.zeros(self.adj_train.shape[0])
        self.arr_deg_full = np.zeros(self.adj_full.shape[0])
        _1 = np.unique(self.adj_train.nonzero()[0],return_counts=True)
        _2 = np.unique(self.adj_full.nonzero()[0],return_counts=True)
        self.arr_deg_train[_1[0]] = _1[1]
        self.arr_deg_full[_2[0]] = _2[1]
        self.avg_deg_train = self.arr_deg_train.mean()
        self.avg_deg_full = self.arr_deg_full.mean()
        # ========================
        self.preproc(**args_preproc)

    @abc.abstractmethod
    def preproc(self,**kwargs):
        pass

    @abc.abstractmethod
    def par_sample(self,stage,**kwargs):
        pass


class rw_sampling(graph_sampler):
    def __init__(self,adj_train,adj_full,node_train,size_subgraph,args_preproc,size_root,size_depth):
        self.size_root = size_root
        self.size_depth = size_depth
        size_subgraph = size_root*size_depth
        super().__init__(adj_train,adj_full,node_train,size_subgraph,args_preproc)
    def preproc(self,**kwargs):
        pass
    def par_sample(self,stage,**kwargs):
        return cy.sampler_rw_cython(self.adj_train.indptr,self.adj_train.indices,\
                self.node_train,self.size_root,self.size_depth,NUM_PROC,RUN_PER_PROC)

class edge_sampling(graph_sampler):
    def __init__(self,adj_train,adj_full,node_train,size_subgraph,args_preproc):
        self.size_subgraph=size_subgraph
        self.indices_lut=np.zeros(adj_train.indices.shape[0],dtype=np.int32)
        for i in range(adj_train.indptr.shape[0]-1):
            self.indices_lut[adj_train.indptr[i]:adj_train.indptr[i+1]]=i
        super().__init__(adj_train,adj_full,node_train,size_subgraph,args_preproc)
    def preproc(self,**kwargs):
        # TODO: calculate the edge probability q_e = 1 - (1-p_e)^{1/m}
        pass
    def par_sample(self,stage,**kwargs):
        return cy.sampler_edge_cython(self.adj_train.indptr,self.adj_train.indices,\
                self.node_train,self.size_subgraph,self.indices_lut,NUM_PROC,RUN_PER_PROC)

class edge_indp_sampling(graph_sampler):
    def __init__(self,adj_train,adj_full,node_train,num_edges_subgraph,args_preproc,level_approx,is_induced):
        """
        level_approx:       level of approximation for deriving the optimal edge sampling probability
        is_induced:         whether you have an induced step at the end of sampling
        num_edges_subgraph: specify the size of subgraph by the edge budget. NOTE: other samplers specify node budget.
        """
        self.num_edges_subgraph = num_edges_subgraph
        self.size_subgraph = num_edges_subgraph*2       # this may not be true in many cases. But for now just use this.
        self.deg_train = np.array(adj_train.sum(1)).flatten()
        self.adj_train_norm = scipy.sparse.dia_matrix((1/self.deg_train,0),shape=adj_train.shape).dot(adj_train)
        super().__init__(adj_train,adj_full,node_train,self.size_subgraph,args_preproc)
    def preproc(self,**kwargs):
        self.edge_prob = scipy.sparse.csr_matrix((np.zeros(self.adj_train.size),\
                self.adj_train.indices,self.adj_train.indptr),shape=self.adj_train.shape)
        #for u in range(self.adj_train.shape[0]):
        #    for j in range(self.adj_train.indptr[u],self.adj_train.indptr[u+1],1):
        #        self.adj_train.data[j]
        self.edge_prob.data[:] = self.adj_train_norm.data[:]
        _adj_trans = scipy.sparse.csr_matrix.tocsc(self.adj_train_norm)
        self.edge_prob.data += _adj_trans.data      # P_e \propto a_{u,v} + a_{v,u}
        self.edge_prob.data *= 2*self.num_edges_subgraph/self.edge_prob.data.sum()
        # now edge_prob is a symmetric matrix, we only keep the upper triangle part, since adj is assumed to be undirected.
        self.edge_prob_tri = scipy.sparse.triu(self.edge_prob)  # NOTE: in coo format
        #import pdb; pdb.set_trace()
    def _sample(self):
        num_gen = np.random.uniform(size=self.edge_prob_tri.data.size)
        edge_idx_selected = np.where(self.edge_prob_tri.data - num_gen >= 0)[0]
        #edge_binmap_selected = np.zeros(self.edge_prob_tri.data.size).astype(np.bool)
        #edge_binmap_selected[edge_idx_selected] = 1
        edge_selected_row = self.edge_prob_tri.row[edge_idx_selected]
        edge_selected_col = self.edge_prob_tri.col[edge_idx_selected]
        node_subg = set(list(edge_selected_row)+list(edge_selected_col))
        node_subg = sorted(list(node_subg))
        _map = {v:i for i,v in enumerate(node_subg)}
        _map_rev = {i:v for i,v in enumerate(node_subg)}
        _fmap = lambda x: _map[x]
        _fmap_rev = lambda x: _map_rev[x]
        _col_remapped = np.fromiter((_fmap(xi) for xi in edge_selected_col),edge_selected_col.dtype)
        _row_remapped = np.fromiter((_fmap(xi) for xi in edge_selected_row),edge_selected_row.dtype)
        _adj_coo = scipy.sparse.coo_matrix((np.ones(_col_remapped.size),(_row_remapped,_col_remapped)),shape=(len(node_subg),len(node_subg)))
        _adj_csr = _adj_coo.tocsr()
        indices_orig = np.fromiter((_fmap_rev(xi) for xi in _adj_csr.indices),_adj_csr.indices.dtype)
        return _adj_csr.indptr, _adj_csr.indices, indices_orig, _adj_csr.data, np.array(node_subg)

    def par_sample(self,stage,**kwargs):
        ret_indptr = list()
        ret_indices = list()
        ret_indices_orig = list()
        ret_data = list()
        ret_node_subg = list()
        for r in range(NUM_PROC*RUN_PER_PROC):
            _indptr,_indices,_indices_orig,_data,_node_subg = self._sample()
            ret_indptr.append(_indptr)
            ret_indices.append(_indices)
            ret_indices_orig.append(_indices_orig)
            ret_data.append(_data)
            ret_node_subg.append(_node_subg)
        return ret_indptr,ret_indices,ret_indices_orig,ret_data,ret_node_subg
                



class khop_sampling(graph_sampler):
    
    def __init__(self,adj_train,adj_full,node_train,size_subgraph,args_preproc,order):
        self.num_proc = 5
        self.p_dist_train = np.zeros(len(node_train))
        self.order = order
        super().__init__(adj_train,adj_full,node_train,size_subgraph,args_preproc)

    def preproc(self,**kwargs):
        """
        This is actually adj to the power of k
        """
        if self.order > 1:
            _adj_hop = '{}/adj_train_hop{}.npz'.format(FLAGS.data_prefix,self.order)
            _adj_hop = scipy.sparse.load_npz(_adj_hop)
        else:
            _adj_hop = self.adj_train
        self.p_dist_train = np.array([_adj_hop.data[_adj_hop.indptr[v]:_adj_hop.indptr[v+1]].sum() for v in self.node_train], dtype=np.int32)

    def par_sample(self,stage,**kwargs):
        _p_cumsum = np.array(self.p_dist_train).astype(np.int64).cumsum()
        if _p_cumsum[-1] > 2**31-1:
            print('warning: total deg exceeds 2**31')
            _p_cumsum = _p_cumsum.astype(np.float64)
            _p_cumsum /= _p_cumsum[-1]/(2**31-1)
            _p_cumsum = _p_cumsum.astype(np.int32)
        _p_cumsum = _p_cumsum.astype(np.int32)
        return cy.sampler_khop_cython(self.adj_train.indptr,self.adj_train.indices,_p_cumsum,\
            self.node_train,self.size_subgraph,NUM_PROC,RUN_PER_PROC)

       


class frontier_sampling(graph_sampler):

    def __init__(self,adj_train,adj_full,node_train,size_subgraph,args_preproc,size_frontier,order,max_deg=10000):
        self.order = order
        self.p_dist = None
        super().__init__(adj_train,adj_full,node_train,size_subgraph,args_preproc)
        self.size_frontier = size_frontier
        self.deg_full = np.bincount(self.adj_full.nonzero()[0])
        self.deg_train = np.bincount(self.adj_train.nonzero()[0])
        self.name_sampler = 'FRONTIER'
        self.max_deg = int(max_deg)

    def preproc(self,**kwargs):
        if self.order > 1:
            f_data = '{}/adj_train_hop{}.npz'.format(FLAGS.data_prefix,self.order)
            _adj_hop = scipy.sparse.load_npz(f_data)
            print('order > 1')
        else:
            _adj_hop = self.adj_train
            print('order == 1')
        # TODO: check, this p_dist shouldn't be normalized. 
        # TODO: check that val and test nodes are 0 for amazon
        self.p_dist = np.array([_adj_hop.data[_adj_hop.indptr[v]:_adj_hop.indptr[v+1]].sum() for v in range(_adj_hop.shape[0])], dtype=np.int32)

    def par_sample(self,stage,**kwargs):
        return cy.sampler_frontier_cython(self.adj_train.indptr,self.adj_train.indices,self.p_dist,\
            self.node_train,self.max_deg,self.size_frontier,self.size_subgraph,NUM_PROC,RUN_PER_PROC)


