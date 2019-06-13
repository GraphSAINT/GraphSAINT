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
                self.node_train,self.size_root,self.size_depth,True,NUM_PAR_SAMPLER,SAMPLES_PER_PROC)

class edge_indp_sampling(graph_sampler):
    def __init__(self,adj_train,adj_full,node_train,num_edges_subgraph,args_preproc):
        """
        num_edges_subgraph: specify the size of subgraph by the edge budget. NOTE: other samplers specify node budget.
        """
        self.num_edges_subgraph = num_edges_subgraph
        self.size_subgraph = num_edges_subgraph*2       # this may not be true in many cases. But for now just use this.
        self.deg_train = np.array(adj_train.sum(1)).flatten()
        self.adj_train_norm = scipy.sparse.dia_matrix((1/self.deg_train,0),shape=adj_train.shape).dot(adj_train)
        super().__init__(adj_train,adj_full,node_train,self.size_subgraph,args_preproc)
        self.level_approx = level_approx
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
        node_binmap = np.zeros(self.adj_train.shape[0]).astype(np.bool)
        node_binmap[node_subg] = 1
        _edge_row = list()
        _edge_col = list()
        for u in node_subg:
            for iv in range(self.adj_train.indptr[u],self.adj_train.indptr[u+1],1):
                if not node_binmap[self.adj_train.indices[iv]]:
                    continue
                _edge_row.append(u)
                _edge_col.append(self.adj_train.indices[iv])
        edge_selected_row = np.array(_edge_row)
        edge_selected_col = np.array(_edge_col)
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
        for r in range(NUM_PAR_SAMPLER*SAMPLES_PER_PROC):
            _indptr,_indices,_indices_orig,_data,_node_subg = self._sample()
            ret_indptr.append(_indptr)
            ret_indices.append(_indices)
            ret_indices_orig.append(_indices_orig)
            ret_data.append(_data)
            ret_node_subg.append(_node_subg)
        return ret_indptr,ret_indices,ret_indices_orig,ret_data,ret_node_subg
                



class frontier_sampling(graph_sampler):

    def __init__(self,adj_train,adj_full,node_train,size_subgraph,args_preproc,size_frontier,max_deg=10000):
        self.p_dist = None
        super().__init__(adj_train,adj_full,node_train,size_subgraph,args_preproc)
        self.size_frontier = size_frontier
        self.deg_full = np.bincount(self.adj_full.nonzero()[0])
        self.deg_train = np.bincount(self.adj_train.nonzero()[0])
        self.name_sampler = 'FRONTIER'
        self.max_deg = int(max_deg)

    def preproc(self,**kwargs):
        _adj_hop = self.adj_train
        self.p_dist = np.array([_adj_hop.data[_adj_hop.indptr[v]:_adj_hop.indptr[v+1]].sum() for v in range(_adj_hop.shape[0])], dtype=np.int32)

    def par_sample(self,stage,**kwargs):
        return cy.sampler_frontier_cython(self.adj_train.indptr,self.adj_train.indices,self.p_dist,\
            self.node_train,self.max_deg,self.size_frontier,self.size_subgraph,NUM_PAR_SAMPLER,SAMPLES_PER_PROC)


