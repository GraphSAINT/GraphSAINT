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
    PIPESIZE = 8000
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
    def sample(self,output_queue,tid,stage,**kwargs):
        assert stage in ['train','val','test']

    @abc.abstractmethod
    def par_sample(self,stage,**kwargs):
        pass

    def fill_pipe(self,output_queue,tid,node_subgraph):
        node_subgraph.sort()
        if output_queue is not None:
            num_seg = int(math.ceil(len(node_subgraph)/self.PIPESIZE))
            for i in reversed(range(num_seg)):
                output_queue.put([tid,i,node_subgraph[i*self.PIPESIZE:(i+1)*self.PIPESIZE]])
        else:
            self.node_subgraph = node_subgraph
            return node_subgraph


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
        pass
    def par_sample(self,stage,**kwargs):
        return cy.sampler_edge_cython(self.adj_train.indptr,self.adj_train.indices,\
                self.node_train,self.size_subgraph,self.indices_lut,NUM_PROC,RUN_PER_PROC)


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

    def sample(self,output_queue,tid,stage):
        assert stage == 'train'
        node_subgraph = random.choices(self.node_train,k=self.size_subgraph,weights=self.p_dist_train)
        node_subgraph = list(set(node_subgraph))
        return self.fill_pipe(output_queue,tid,node_subgraph)
        


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

    def double_arr_indicator(self, arr_indicator,k=2):
        a = np.zeros((k*arr_indicator.size,2),dtype=arr_indicator.dtype)
        a[:arr_indicator.shape[0]] = arr_indicator
        return a

    def par_sample(self,stage,**kwargs):
        return cy.sampler_frontier_cython(self.adj_train.indptr,self.adj_train.indices,self.p_dist,\
            self.node_train,self.max_deg,self.size_frontier,self.size_subgraph,NUM_PROC,RUN_PER_PROC)


    def sample(self,output_queue,tid,stage,frontier=None):
        assert stage == 'train'     # higher order frontier now only support training stage
        m = self.size_frontier
        super().sample(output_queue,tid,stage)
        t1 = time.time()
        assert m is not None or frontier is not None
        #EPSILON = 1e-5
        assert frontier is None
        if frontier is None:
            frontier = random.choices(self.node_train,k=m)
            #np.random.choice(np.arange(self.adj_full.shape[0]),m,p=_prob)
        else:
            m = frontier.size
        _adj = self.adj_train if stage=='train' else self.adj_full
        #_deg = self.deg_train if stage=='train' else self.deg_full
        lambd = 0
        # TODO: debug here
        #_arr_deg_ = self.arr_deg_train if stage=='train' else self.arr_deg_full
        _arr_deg = self.p_dist
        _arr_deg = np.clip(_arr_deg,0,self.max_deg) + lambd
        _avg_deg = _arr_deg.mean()
        node_subgraph = []
        # ======================================================
        alpha = 3
        # ---------------------
        # arr_indicator:
        # suppose 2 nodes, u w/ deg 3 and v w/ deg 4
        # arr_indicator is a (7,2) shape array
        #  u  u  u  v  v  v  v
        # -3  1  2 -4  1  2  3
        #  1  1  1  1  2  2  2
        arr_indicator = np.zeros((int(alpha*_avg_deg*m),2),dtype=np.int64)
        # TODO: max deg is not enforced here
        deg_cumsum = np.cumsum([0]+[_arr_deg[f] for f in frontier]).astype(np.int64)
        end_idx = deg_cumsum[-1]
        if end_idx > arr_indicator.shape[0]:
            print('doubling')
            arr_indicator = self.double_arr_indicator(arr_indicator,k=ceil(end_idx/arr_indicator.shape[0]))
        for i,mi in enumerate(frontier):
            arr_indicator[deg_cumsum[i]:deg_cumsum[i+1],0] = mi
            arr_indicator[deg_cumsum[i]:deg_cumsum[i+1],1] = np.arange(deg_cumsum[i+1]-deg_cumsum[i])
            arr_indicator[deg_cumsum[i],1] = deg_cumsum[i] - deg_cumsum[i+1]
 
        for cur_size in range(m,self.size_subgraph+1):
            while True:
                idx = random.randint(0,end_idx-1)
                if arr_indicator[idx,0]:    # TODO: arr_indicator[idx,0] stores the vertex id or 0. However, what about the vertex id is 0?
                    break
            selected_v,offset = arr_indicator[idx]
            # *********************
            idx = idx if offset<0 else idx-offset
            idx = int(idx)
            offset = -1*(offset if offset<0 else arr_indicator[idx,1])
            offset = int(offset)
            arr_indicator[idx:idx+offset] = 0
            #neighs = _adj.indices[_adj.indptr[selected_v]:_adj.indptr[selected_v+1]]
            num_neighs = _adj.indptr[selected_v+1]-_adj.indptr[selected_v]
            #new_frontier = neighs[random.randint(0,neighs.size-1)]
            new_frontier = _adj.indices[_adj.indptr[selected_v]+random.randint(0,num_neighs-1)]
            node_subgraph.append(new_frontier)
            # TODO: cython up to here
            _deg = min(self.p_dist[new_frontier],self.max_deg)+lambd
            if end_idx+_deg > arr_indicator.shape[0]:
                # shift arr_indicator to fill in the gaps of 0
                _start = np.where(arr_indicator[:,1]<0)[0].astype(np.int64)
                _end = _start-arr_indicator[_start,1]
                _end[1:] = _end[:-1]
                _end[0] = 0
                delta = _start-_end
                delta = np.cumsum(delta).astype(np.int64)
                end_idx = (-arr_indicator[_start,1]).sum().astype(np.int64)
                for i,stepi in enumerate(delta):
                    _s = _start[i]
                    _e = _start[i]-arr_indicator[_start[i],1]
                    arr_indicator[_s-stepi:_e-stepi] = arr_indicator[_s:_e]
                if end_idx+_deg > arr_indicator.shape[0]:       # TODO: now here one time doubling may not work, need a while loop
                    print('doubling')
                    arr_indicator = self.double_arr_indicator(arr_indicator,k=ceil((end_idx+_deg)/arr_indicator.shape[0]))
            arr_indicator[end_idx:end_idx+_deg,0] = new_frontier
            arr_indicator[end_idx:end_idx+_deg,1] = np.arange(_deg)
            arr_indicator[end_idx,1] = -_deg
            end_idx += _deg

        # ======================================================
        node_subgraph.extend(list(frontier))
        node_subgraph = list(set(node_subgraph))
        t2 = time.time()
        return self.fill_pipe(output_queue,tid,node_subgraph)



class community_sampling(graph_sampler):

    def __init__(self,adj_train,adj_full,node_train,size_subgraph):
        super().__init__(adj_train,adj_full,node_train,size_subgraph)
        self.classloaded = 0
        self.class_map_dict = dict()
        self.name_sampler = 'COMMUNITY'

    def sample(self,stage):
        super().sample(stage)
        _adj = self.adj_train if stage=='train' else self.adj_full
        if self.classloaded == 0:
            self.class_map_dict = json.load(open(prefix+'/class_map.json'))
            self.classloaded = 1
        num_vertices = len(set(_adj.nonzero()[0]))
        classified = 0;
        curr_class = 0;
        budget = self.size_subgraph / num_vertices
        node_subgraph = [];
        while classified < num_vertices:
            curr_class_set = [int(k) for k,v in self.class_map_dict.items() if v==curr_class and len(_adj[int(k)].nonzero()[0])>0]
            node_subgraph.extend(random.sample(curr_class_set,int(len(curr_class_set)*budget)))
            classified += len(curr_class_set)
            curr_class += 1
        printf('subgraph size ={}',len(set(node_subgraph)))
        self.node_subgraph = np.array(node_subgraph)
        return self.node_subgraph
