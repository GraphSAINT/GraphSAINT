from graphsaint.globals import *
import math
from graphsaint.tensorflow_version.inits import *
from graphsaint.utils import *
from graphsaint.graph_samplers import *
from graphsaint.norm_aggr import *
import tensorflow as tf
import scipy.sparse as sp
import scipy

import numpy as np
import time

import pdb




class Minibatch:
    """
    This minibatch iterator iterates over nodes for supervised learning.
    """

    def __init__(self, adj_full_norm, adj_train, role, class_arr, placeholders, train_params, **kwargs):
        """
        role:       array of string (length |V|)
                    storing role of the node ('tr'/'va'/'te')
        class_arr: array of float (shape |V|xf)
                    storing initial feature vectors
        """
        self.node_train = np.array(role['tr'])
        self.node_val = np.array(role['va'])
        self.node_test = np.array(role['te'])

        self.class_arr = class_arr
        self.adj_full_norm = adj_full_norm
        s1=int(adj_full_norm.shape[0]/8*1)
        s2=int(adj_full_norm.shape[0]/8*2)
        s3=int(adj_full_norm.shape[0]/8*3)
        s4=int(adj_full_norm.shape[0]/8*4)
        s5=int(adj_full_norm.shape[0]/8*5)
        s6=int(adj_full_norm.shape[0]/8*6)
        s7=int(adj_full_norm.shape[0]/8*7)
        self.dim0_adj_sub = adj_full_norm.shape[0]/8
        self.adj_full_norm_0=adj_full_norm[:s1,:]
        self.adj_full_norm_1=adj_full_norm[s1:s2,:]
        self.adj_full_norm_2=adj_full_norm[s2:s3,:]
        self.adj_full_norm_3=adj_full_norm[s3:s4,:]
        self.adj_full_norm_4=adj_full_norm[s4:s5,:]
        self.adj_full_norm_5=adj_full_norm[s5:s6,:]
        self.adj_full_norm_6=adj_full_norm[s6:s7,:]
        self.adj_full_norm_7=adj_full_norm[s7:,:]
        self.adj_train = adj_train

        assert self.class_arr.shape[0] == self.adj_full_norm.shape[0]

        # below: book-keeping for mini-batch
        self.placeholders = placeholders
        self.node_subgraph = None
        self.batch_num = -1

        self.method_sample = None
        self.subgraphs_remaining_indptr = []
        self.subgraphs_remaining_indices = []
        self.subgraphs_remaining_data = []
        self.subgraphs_remaining_nodes = []
        self.subgraphs_remaining_edge_index = []
        
        self.norm_loss_train = np.zeros(self.adj_train.shape[0])
        # norm_loss_test is used in full batch evaluation (without sampling). so neighbor features are simply averaged.
        self.norm_loss_test = np.zeros(self.adj_full_norm.shape[0])
        _denom = len(self.node_train) + len(self.node_val) +  len(self.node_test)
        self.norm_loss_test[self.node_train] = 1./_denom
        self.norm_loss_test[self.node_val] = 1./_denom
        self.norm_loss_test[self.node_test] = 1./_denom
        self.norm_aggr_train = np.zeros(self.adj_train.size)
       
        self.sample_coverage = train_params['sample_coverage']
        self.dropout = train_params['dropout']
        self.deg_train = np.array(self.adj_train.sum(1)).flatten()


    def set_sampler(self,train_phases):
        self.subgraphs_remaining_indptr = list()
        self.subgraphs_remaining_indices = list()
        self.subgraphs_remaining_data = list()
        self.subgraphs_remaining_nodes = list()
        self.subgraphs_remaining_edge_index = list()
        self.method_sample = train_phases['sampler']
        if self.method_sample == 'mrw':
            if 'deg_clip' in train_phases:
                _deg_clip = int(train_phases['deg_clip'])
            else:
                _deg_clip = 100000      # setting this to a large number so essentially there is no clipping in probability
            self.size_subg_budget = train_phases['size_subgraph']
            self.graph_sampler = mrw_sampling(self.adj_train,\
                self.node_train,self.size_subg_budget,train_phases['size_frontier'],_deg_clip)
        elif self.method_sample == 'rw':
            self.size_subg_budget = train_phases['num_root']*train_phases['depth']
            self.graph_sampler = rw_sampling(self.adj_train,\
                self.node_train,self.size_subg_budget,int(train_phases['num_root']),int(train_phases['depth']))
        elif self.method_sample == 'edge':
            self.size_subg_budget = train_phases['size_subg_edge']*2
            self.graph_sampler = edge_sampling(self.adj_train,self.node_train,train_phases['size_subg_edge'])
        elif self.method_sample == 'node':
            self.size_subg_budget = train_phases['size_subgraph']
            self.graph_sampler = node_sampling(self.adj_train,self.node_train,self.size_subg_budget)
        elif self.method_sample == 'full_batch':
            self.size_subg_budget = self.node_train.size
            self.graph_sampler = full_batch_sampling(self.adj_train,self.node_train,self.size_subg_budget)
        else:
            raise NotImplementedError

        self.norm_loss_train = np.zeros(self.adj_train.shape[0])
        self.norm_aggr_train = np.zeros(self.adj_train.size).astype(np.float32)

        # For edge sampler, no need to estimate norm factors, we can calculate directly.
        # However, for integrity of the framework, we decide to follow the same procedure for all samplers: 
        # 1. sample enough number of subgraphs
        # 2. estimate norm factor alpha and lambda
        tot_sampled_nodes = 0
        while True:
            self.par_graph_sample('train')
            tot_sampled_nodes = sum([len(n) for n in self.subgraphs_remaining_nodes])
            if tot_sampled_nodes > self.sample_coverage*self.node_train.size:
                break
        print()
        num_subg = len(self.subgraphs_remaining_nodes)
        for i in range(num_subg):
            self.norm_aggr_train[self.subgraphs_remaining_edge_index[i]] += 1
            self.norm_loss_train[self.subgraphs_remaining_nodes[i]] += 1
        assert self.norm_loss_train[self.node_val].sum() + self.norm_loss_train[self.node_test].sum() == 0
        for v in range(self.adj_train.shape[0]):
            i_s = self.adj_train.indptr[v]
            i_e = self.adj_train.indptr[v+1]
            val = np.clip(self.norm_loss_train[v]/self.norm_aggr_train[i_s:i_e], 0, 1e4)
            val[np.isnan(val)] = 0.1
            self.norm_aggr_train[i_s:i_e] = val
        self.norm_loss_train[np.where(self.norm_loss_train==0)[0]] = 0.1
        self.norm_loss_train[self.node_val] = 0
        self.norm_loss_train[self.node_test] = 0
        self.norm_loss_train[self.node_train] = num_subg/self.norm_loss_train[self.node_train]/self.node_train.size

    def par_graph_sample(self,phase):
        t0 = time.time()
        _indptr,_indices,_data,_v,_edge_index= self.graph_sampler.par_sample(phase)
        t1 = time.time()
        print('sampling 200 subgraphs:   time = {:.3f} sec'.format(t1-t0), end="\r")
        self.subgraphs_remaining_indptr.extend(_indptr)
        self.subgraphs_remaining_indices.extend(_indices)
        self.subgraphs_remaining_data.extend(_data)
        self.subgraphs_remaining_nodes.extend(_v)
        self.subgraphs_remaining_edge_index.extend(_edge_index)

    def feed_dict(self,mode='train'):
        """ DONE """
        if mode in ['val','test']:
            self.node_subgraph = np.arange(self.class_arr.shape[0])
            adj = sp.csr_matrix(([],[],np.zeros(2)), shape=(1,self.node_subgraph.shape[0]))
            #adj = self.adj_full_norm
            adj_0 = self.adj_full_norm_0
            adj_1 = self.adj_full_norm_1
            adj_2 = self.adj_full_norm_2
            adj_3 = self.adj_full_norm_3
            adj_4 = self.adj_full_norm_4
            adj_5 = self.adj_full_norm_5
            adj_6 = self.adj_full_norm_6
            adj_7 = self.adj_full_norm_7
            _dropout = 0.
        else:
            assert mode == 'train'
            tt0=time.time()
            if len(self.subgraphs_remaining_nodes) == 0:
                self.par_graph_sample('train')
                print()

            self.node_subgraph = self.subgraphs_remaining_nodes.pop()
            self.size_subgraph = len(self.node_subgraph)
            adj = sp.csr_matrix((self.subgraphs_remaining_data.pop(),self.subgraphs_remaining_indices.pop(),\
                        self.subgraphs_remaining_indptr.pop()),shape=(self.node_subgraph.size,self.node_subgraph.size))
            adj_edge_index=self.subgraphs_remaining_edge_index.pop()
            #print("{} nodes, {} edges, {} degree".format(self.node_subgraph.size,adj.size,adj.size/self.node_subgraph.size))
            tt1 = time.time()
            assert len(self.node_subgraph) == adj.shape[0]
            norm_aggr(adj.data,adj_edge_index,self.norm_aggr_train,num_proc=args_global.num_cpu_core)

            tt2 = time.time()
            adj = adj_norm(adj, deg=self.deg_train[self.node_subgraph])

            adj_0 = sp.csr_matrix(([],[],np.zeros(2)),shape=(1,self.node_subgraph.shape[0]))
            adj_1 = sp.csr_matrix(([],[],np.zeros(2)),shape=(1,self.node_subgraph.shape[0]))
            adj_2 = sp.csr_matrix(([],[],np.zeros(2)),shape=(1,self.node_subgraph.shape[0]))
            adj_3 = sp.csr_matrix(([],[],np.zeros(2)),shape=(1,self.node_subgraph.shape[0]))
            adj_4 = sp.csr_matrix(([],[],np.zeros(2)),shape=(1,self.node_subgraph.shape[0]))
            adj_5 = sp.csr_matrix(([],[],np.zeros(2)),shape=(1,self.node_subgraph.shape[0]))
            adj_6 = sp.csr_matrix(([],[],np.zeros(2)),shape=(1,self.node_subgraph.shape[0]))
            adj_7 = sp.csr_matrix(([],[],np.zeros(2)),shape=(1,self.node_subgraph.shape[0]))

            _dropout = self.dropout

            self.batch_num += 1
        feed_dict = dict()
        feed_dict.update({self.placeholders['node_subgraph']: self.node_subgraph})
        feed_dict.update({self.placeholders['labels']: self.class_arr[self.node_subgraph]})
        feed_dict.update({self.placeholders['dropout']: _dropout})
        if mode in ['val','test']:
            feed_dict.update({self.placeholders['norm_loss']: self.norm_loss_test})
        else:
            feed_dict.update({self.placeholders['norm_loss']: self.norm_loss_train})
        
        _num_edges = len(adj.nonzero()[1])
        _num_vertices = len(self.node_subgraph)
        _indices_ph = np.column_stack(adj.nonzero())
        _shape_ph = adj.shape
        feed_dict.update({self.placeholders['adj_subgraph']: \
            tf.SparseTensorValue(_indices_ph,adj.data,_shape_ph)})
        feed_dict.update({self.placeholders['adj_subgraph_0']: \
            tf.SparseTensorValue(np.column_stack(adj_0.nonzero()),adj_0.data,adj_0.shape)})
        feed_dict.update({self.placeholders['adj_subgraph_1']: \
            tf.SparseTensorValue(np.column_stack(adj_1.nonzero()),adj_1.data,adj_1.shape)})
        feed_dict.update({self.placeholders['adj_subgraph_2']: \
            tf.SparseTensorValue(np.column_stack(adj_2.nonzero()),adj_2.data,adj_2.shape)})
        feed_dict.update({self.placeholders['adj_subgraph_3']: \
            tf.SparseTensorValue(np.column_stack(adj_3.nonzero()),adj_3.data,adj_3.shape)})
        feed_dict.update({self.placeholders['adj_subgraph_4']: \
            tf.SparseTensorValue(np.column_stack(adj_4.nonzero()),adj_4.data,adj_4.shape)})
        feed_dict.update({self.placeholders['adj_subgraph_5']: \
            tf.SparseTensorValue(np.column_stack(adj_5.nonzero()),adj_5.data,adj_5.shape)})
        feed_dict.update({self.placeholders['adj_subgraph_6']: \
            tf.SparseTensorValue(np.column_stack(adj_6.nonzero()),adj_6.data,adj_6.shape)})
        feed_dict.update({self.placeholders['adj_subgraph_7']: \
            tf.SparseTensorValue(np.column_stack(adj_7.nonzero()),adj_7.data,adj_7.shape)})
        feed_dict.update({self.placeholders['dim0_adj_sub']:\
            self.dim0_adj_sub})
        tt3=time.time()
        # if mode in ['train']:
        #     print("t1:{:.3f} t2:{:.3f} t3:{:.3f}".format(tt0-tt1,tt2-tt1,tt3-tt2))
        if mode in ['val','test']:
            feed_dict[self.placeholders['is_train']]=False
        else:
            feed_dict[self.placeholders['is_train']]=True
        return feed_dict, self.class_arr[self.node_subgraph]


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
