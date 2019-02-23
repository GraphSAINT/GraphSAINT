from graphsaint.globals import *
import math
from graphsaint.inits import *
from graphsaint.utils import *
from graphsaint.graph_samplers import *
from graphsaint.adj_misc import *
import tensorflow as tf
import scipy.sparse as sp
import pandas as pd

import numpy as np
import time
import multiprocessing as mp

import pdb

# import warnings
# warnings.filterwarnings('error')
# np.seterr(divide='raise')

#np.random.seed(123)


############################
# FOR SUPERVISED MINIBATCH #
############################
class NodeMinibatchIterator(object):
    """
    This minibatch iterator iterates over nodes for supervised learning.
    """

    def __init__(self, adj_full, adj_full_norm, adj_train, role, class_arr, placeholders, train_params, **kwargs):
        """
        role:       array of string (length |V|)
                    storing role of the node ('tr'/'va'/'te')
        class_arr: array of float (shape |V|xf)
                    storing initial feature vectors
        TODO:       adj_full_norm: normalized adj. Norm by # non-zeros per row
        """
        self.num_proc = 1
        self.node_train = np.array(role['tr'])
        self.node_val = np.array(role['va'])
        self.node_test = np.array(role['te'])

        self.class_arr = class_arr
        self.adj_full = adj_full
        self.adj_full_norm = adj_full_norm
        self.adj_train = adj_train
        self.adj_full_rev = self.adj_full.transpose()
        self.adj_train_rev = self.adj_train.transpose()

        assert self.class_arr.shape[0] == self.adj_full.shape[0]

        # below: book-keeping for mini-batch
        self.placeholders = placeholders
        self.node_subgraph = None
        self.batch_num = -1

        self.method_sample = None
        self.subgraphs_remaining_indptr = []
        self.subgraphs_remaining_indices = []
        self.subgraphs_remaining_data = []
        self.subgraphs_remaining_nodes = []
        
        self.norm_weight_train = None
        self.norm_weight_test = np.zeros(self.adj_full.shape[0])
        self.norm_weight_test[self.node_train] = 1
        self.norm_weight_test[self.node_val] = 1
        self.norm_weight_test[self.node_test] = 1
        self.norm_adj = train_params['norm_adj']
        self.q_threshold = train_params['q_threshold']
        self.q_offset = train_params['q_offset']

    def _set_sampler_args(self):
        if self.method_sample == 'frontier':
            _args = {'frontier':None}
        elif self.method_sample == 'khop':
            _args = dict()
        elif self.method_sample == 'rw':
            _args = dict()
        elif self.method_sample == 'edge':
            _args = dict()
        else:
            raise NotImplementedError
        return _args


    def set_sampler(self,train_phases,is_norm_weight=False):
        self.subgraphs_remaining_indptr = list()
        self.subgraphs_remaining_indices = list()
        self.subgraphs_remaining_data = list()
        self.subgraphs_remaining_nodes = list()
        self.method_sample = train_phases['sampler']
        if self.method_sample == 'frontier':
            self.size_subg_budget = train_phases['size_subgraph']
            self.graph_sampler = frontier_sampling(self.adj_train,self.adj_full,\
                self.node_train,self.size_subg_budget,dict(),train_phases['size_frontier'],int(train_phases['order']),int(train_phases['max_deg']))
        elif self.method_sample == 'khop':
            self.size_subg_budget = train_phases['size_subgraph']
            self.graph_sampler = khop_sampling(self.adj_train,self.adj_full,\
                self.node_train,self.size_subg_budget,dict(),int(train_phases['order']))
        elif self.method_sample == 'rw':
            self.size_subg_budget = train_phases['num_root']*train_phases['depth']
            self.graph_sampler = rw_sampling(self.adj_train,self.adj_full,\
                self.node_train,self.size_subg_budget,dict(),int(train_phases['num_root']),int(train_phases['depth']))
        elif self.method_sample == 'edge':
            self.size_subg_budget = train_phases['size_subgraph']
            self.graph_sampler = edge_sampling(self.adj_train,self.adj_full,self.node_train,\
                                               self.size_subg_budget,dict())
        else:
            raise NotImplementedError

        self.norm_weight_train = np.zeros(self.adj_full.shape[0])
        if not is_norm_weight:
            self.norm_weight_train[self.node_train] = 1
        else:
            subg_end = 0; tot_sampled_nodes = 0
            while True:
                self.par_graph_sample('train')
                for i in range(subg_end,len(self.subgraphs_remaining_nodes)):
                    self.norm_weight_train[self.subgraphs_remaining_nodes[i]] += 1
                    tot_sampled_nodes += self.subgraphs_remaining_nodes[i].size
                if tot_sampled_nodes > self.q_threshold*self.node_train.size:
                    break
                subg_end = len(self.subgraphs_remaining_nodes)
            self.norm_weight_train[self.node_train] += self.q_offset
            if self.norm_weight_train[self.node_train].min() == 0:
                self.norm_weight_train[self.node_train] += 1
            self.norm_weight_train[self.node_train] = 1/self.norm_weight_train[self.node_train]
            avg_subg_size = tot_sampled_nodes/len(self.subgraphs_remaining_nodes)
            self.norm_weight_train = self.norm_weight_train\
                        /self.norm_weight_train.sum()*self.node_train.size

    def par_graph_sample(self,phase):
        t0 = time.time()
        _indptr,_indices,_data,_v = self.graph_sampler.par_sample(phase,**self._set_sampler_args())
        t1 = time.time()
        print('sampling 200 subgraphs:   time = ',t1-t0)
        self.subgraphs_remaining_indptr.extend(_indptr)
        self.subgraphs_remaining_indices.extend(_indices)
        self.subgraphs_remaining_data.extend(_data)
        self.subgraphs_remaining_nodes.extend(_v)

    def minibatch_train_feed_dict(self,dropout,is_val=False,is_test=False):
        """ DONE """
        if is_val or is_test:
            self.node_subgraph = np.arange(self.class_arr.shape[0])
            adj = self.adj_full_norm
        else:
            if len(self.subgraphs_remaining_nodes) == 0:
                self.par_graph_sample('train')

            self.node_subgraph = self.subgraphs_remaining_nodes.pop()
            self.size_subgraph = len(self.node_subgraph)
            adj = sp.csr_matrix((self.subgraphs_remaining_data.pop(),self.subgraphs_remaining_indices.pop(),\
                        self.subgraphs_remaining_indptr.pop()),shape=(self.node_subgraph.size,self.node_subgraph.size))
            adj = adj_norm(adj,self.norm_adj)
            self.batch_num += 1
        feed_dict = dict()
        feed_dict.update({self.placeholders['node_subgraph']: self.node_subgraph})
        feed_dict.update({self.placeholders['labels']: self.class_arr[self.node_subgraph]})
        feed_dict.update({self.placeholders['dropout']: dropout})
        if is_val or is_test:
            feed_dict.update({self.placeholders['norm_weight']: self.norm_weight_test})
        else:
            feed_dict.update({self.placeholders['norm_weight']:self.norm_weight_train})
        
        _num_edges = len(adj.nonzero()[1])
        _num_vertices = len(self.node_subgraph)
        _indices_ph = np.column_stack(adj.nonzero())
        _shape_ph = adj.shape
        feed_dict.update({self.placeholders['adj_subgraph']: \
            tf.SparseTensorValue(_indices_ph,adj.data,_shape_ph)})
        feed_dict.update({self.placeholders['nnz']: adj.size})
        if is_val or is_test:
            feed_dict[self.placeholders['is_train']]=False
        else:
            feed_dict[self.placeholders['is_train']]=True
        return feed_dict, self.class_arr[self.node_subgraph]


    # dont use this
    def par_graph_sample_multiprocess(self,stage,prefix,args_dict):
        if self.num_proc > 1:
            printf('par sampling',type='WARN')
            output_subgraphs = mp.Queue()
            processes = [mp.Process(target=self.graph_sampler.sample,args=(output_subgraphs,i,stage),kwargs=args_dict) for i in range(self.num_proc)]
            for p in processes:
                p.start()
            num_proc_done = 0
            subgraph_assembler = dict()
            for i in range(self.num_proc):
                subgraph_assembler[i] = []
            while True:
                seg = output_subgraphs.get()
                subgraph_assembler[seg[0]].extend(seg[2])
                if seg[1] == 0:
                    num_proc_done += 1
                if num_proc_done == self.num_proc:
                    break
            for p in processes:
                p.join()
            for k,v in subgraph_assembler.items():
                self.subgraphs_remaining.append(np.array(v))
            #self.subgraphs_remaining.extend([output_subgraphs.get() for p in processes])
        else:
            ret = self.graph_sampler.sample(None,0,stage,**args_dict)
            self.subgraphs_remaining.append(np.array(ret))



    def num_training_batches(self):
        """ DONE """
        return math.ceil(self.node_train.shape[0]/float(self.size_subg_budget))

    def shuffle(self):
        """ DONE """
        self.node_train = np.random.permutation(self.node_train)
        self.batch_num = -1

    def end(self):
        """ DONE """
        return (self.batch_num+1)*self.size_subg_budget >= self.node_train.shape[0]
