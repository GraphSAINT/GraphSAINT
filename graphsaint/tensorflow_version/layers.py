import tensorflow as tf
from graphsaint.tensorflow_version.inits import glorot,zeros,trained,ones,xavier,uniform
from graphsaint.globals import *

F_ACT = {'I': lambda x:x,
         'relu': tf.nn.relu,
         'leaky_relu': tf.nn.leaky_relu}


# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}

def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


class Layer:
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).
    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging', 'mulhead'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging:
                if type(inputs)==type([]) or type(inputs)==type((1,2)):
                    _ip = inputs[0]
                else:
                    _ip = inputs
                tf.summary.histogram(self.name + '/inputs', _ip)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
        return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])



class JumpingKnowledge(Layer):
    def __init__(self, arch_gcn, dim_input_jk, mode=None, **kwargs):
        """
        """
        super(JumpingKnowledge,self).__init__(**kwargs)
        self.mode = mode
        if not mode:
            return
        self.act = F_ACT[arch_gcn['act']]
        self.bias = arch_gcn['bias']
        self.dim_in = dim_input_jk
        self.dim_out = arch_gcn['dim']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([self.dim_in,self.dim_out],name='weights')
            self.vars['bias'] = zeros([self.dim_out],name='bias')
            if self.bias == 'norm':
                self.vars['offset'] = zeros([1,self.dim_out],name='offset')
                self.vars['scale'] = ones([1,self.dim_out],name='scale')
        

    def _call(self, inputs):
        feats_l,idx_conv = inputs
        if not self.mode:
            return feats_l[-1]
        elif self.mode == 'concat':
            feats_sel = [f for i,f in enumerate(feats_l) if i in idx_conv]
            feats_aggr = tf.concat(feats_sel, axis=1)
        elif self.mode == 'max_pool':
            feats_sel = [f for i,f in enumerate(feats_l) if i in idx_conv]
            feats_stack = tf.stack(feats_sel)
            feats_aggr =  tf.reduce_max(feats_stack,axis=0)
        else:
            raise NotImplementedError
        vw = tf.matmul(feats_aggr,self.vars['weights'])
        vw += self.vars['bias']
        vw = self.act(vw)
        if self.bias == 'norm':
            mean,variance = tf.nn.moments(vw,axes=[1],keep_dims=True)
            vw = tf.nn.batch_normalization(vw,mean,variance,self.vars['offset'],self.vars['scale'],1e-9)
        return vw




class HighOrderAggregator(Layer):
    """
    If order == 1, then this layer is the normal GCN layer. If order == 0, this layer is equivalent to a dense layer (only self-to-self propagation).
    If order > 1, then this layer is a high-order layer propagating multi-hop information.
    """
    def __init__(self, dim_in, dim_out, dropout=0., act='relu', \
            order=1, aggr='mean', is_train=True, bias='norm', **kwargs):
        super(HighOrderAggregator,self).__init__(**kwargs)
        self.dropout = dropout
        self.bias = bias
        self.act = F_ACT[act]
        self.order = order
        self.aggr = aggr
        self.is_train = is_train
        if dim_out > 0:
            with tf.variable_scope(self.name + '_vars'):
                for o in range(self.order+1):
                    _k = 'order{}_weights'.format(o)
                    self.vars[_k] = glorot([dim_in,dim_out],name=_k)
                for o in range(self.order+1):
                    _k = 'order{}_bias'.format(o)
                    self.vars[_k] = zeros([dim_out],name=_k)
                if self.bias == 'norm':
                    for o in range(self.order+1):
                        _k1 = 'order{}_offset'.format(o)
                        _k2 = 'order{}_scale'.format(o)
                        self.vars[_k1] = zeros([1,dim_out],name=_k1)
                        self.vars[_k2] = ones([1,dim_out],name=_k2)
        print('>> layer {}, dim: [{},{}]'.format(self.name, dim_in, dim_out))
        if self.logging:
            self._log_vars()

        self.dim_in = dim_in
        self.dim_out = dim_out


    def _F_nonlinear(self,vecs,order):
        vw = tf.matmul(vecs,self.vars['order{}_weights'.format(order)])
        vw += self.vars['order{}_bias'.format(order)]
        vw = self.act(vw)
        if self.bias == 'norm':   # batch norm realized by tf.nn.batch_norm (consistent with SGCN implementation)
            mean,variance = tf.nn.moments(vw,axes=[1],keep_dims=True)
            _off = 'order{}_offset'.format(order)
            _sca = 'order{}_scale'.format(order)
            vw = tf.nn.batch_normalization(vw,mean,variance,self.vars[_off],self.vars[_sca],1e-9)
        return vw

    def _call(self, inputs):
        # vecs: input feature of the current layer. 
        # adj_partition_list: the row partitions of the full graph adj 
        #       (only used in full-batch evaluation on the val/test sets)
        vecs, adj_norm, len_feat, adj_partition_list, _ = inputs
        vecs = tf.nn.dropout(vecs, 1-self.dropout)
        vecs_hop = [tf.identity(vecs) for o in range(self.order+1)]
        for o in range(self.order):
            for a in range(o+1):
                ans1 = tf.sparse_tensor_dense_matmul(adj_norm,vecs_hop[o+1])
                ans_partition = [tf.sparse_tensor_dense_matmul(adj,vecs_hop[o+1]) for adj in adj_partition_list]
                ans2 = tf.concat(ans_partition,0)
                vecs_hop[o+1]=tf.cond(self.is_train,lambda: tf.identity(ans1),lambda: tf.identity(ans2))
        vecs_hop = [self._F_nonlinear(v,o) for o,v in enumerate(vecs_hop)]    
        if self.aggr == 'mean':
            ret = vecs_hop[0]
            for o in range(len(vecs_hop)-1):
                ret += vecs_hop[o+1]
        elif self.aggr == 'concat':
            ret = tf.concat(vecs_hop,axis=1)
        else:
            raise NotImplementedError
        return ret


class AttentionAggregator(Layer):
    """
    Attention mechanism by GAT. We remove the softmax step since during minibatch training, we cannot see all neighbors of a node.
    """
    def __init__(self, dim_in, dim_out,
            dropout=0., act='relu', order=1, aggr='mean', is_train=True, bias='norm', **kwargs):
        assert order <= 1, "now only support attention for order 0/1 layers"
        super(AttentionAggregator,self).__init__(**kwargs)
        self.dropout = dropout
        self.bias = bias
        self.act = F_ACT[act]
        self.order = order
        self.aggr = aggr
        self.is_train = is_train
        if 'mulhead' in kwargs.keys():
            self.mulhead = int(kwargs['mulhead'])
        else:
            self.mulhead = 1
        with tf.variable_scope(self.name + '_vars'):
            self.vars['order0_weights'] = glorot([dim_in,dim_out],name='order0_weights')
            for k in range(self.mulhead):
                self.vars['order1_weights_h{}'.format(k)] = glorot([dim_in,int(dim_out/self.mulhead)],name='order1_weights_h{}'.format(k))
            self.vars['order0_bias'] = zeros([dim_out],name='order0_bias')
            for k in range(self.mulhead):
                self.vars['order1_bias_h{}'.format(k)] = zeros([int(dim_out/self.mulhead)],name='order1_bias_h{}'.format(k))

            if self.bias == 'norm':
                for o in range(self.order+1):
                    _k1 = 'order{}_offset'.format(o)
                    _k2 = 'order{}_scale'.format(o)
                    self.vars[_k1] = zeros([1,dim_out],name=_k1)
                    self.vars[_k2] = ones([1,dim_out],name=_k2)
            for k in range(self.mulhead):
                self.vars['attention_0_h{}'.format(k)] = glorot([1,int(dim_out/self.mulhead)],name='attention_0_h{}'.format(k))
                self.vars['attention_1_h{}'.format(k)] = glorot([1,int(dim_out/self.mulhead)],name='attention_1_h{}'.format(k))
                self.vars['att_bias_0_h{}'.format(k)] = zeros([1],name='att_bias_0_h{}'.format(k))
                self.vars['att_bias_1_h{}'.format(k)] = zeros([1],name='att_bias_1_h{}'.format(k))
        print('>> layer {}, dim: [{},{}]'.format(self.name, dim_in, dim_out))
        if self.logging:
            self._log_vars()

        self.dim_in = dim_in
        self.dim_out = dim_out

    def _F_edge_weight(self,adj_part,vecs_neigh,vecs_self,offset=0):
        adj_mask = tf.dtypes.cast(tf.dtypes.cast(adj_part, tf.bool), tf.float32)
        a1 = tf.SparseTensor(adj_mask.indices,tf.nn.embedding_lookup(vecs_neigh,adj_mask.indices[:,1]),adj_mask.dense_shape)
        a2 = tf.SparseTensor(adj_mask.indices,tf.nn.embedding_lookup(vecs_self,adj_mask.indices[:,0]+offset),adj_mask.dense_shape)
        alpha = tf.SparseTensor(adj_mask.indices,tf.nn.relu(a1.values+a2.values),adj_mask.dense_shape)
        adj_weighted = tf.SparseTensor(adj_mask.indices,adj_part.values*alpha.values,adj_mask.dense_shape)
        return adj_weighted


    def _call(self, inputs):
    
        vecs, adj_norm, len_feat, adj_partition_list, dim0_adj_sub = inputs
        adj_norm = tf.cond(self.is_train,lambda: adj_norm,lambda: tf.sparse.concat(0,adj_partition_list))
        vecs_do1 = tf.nn.dropout(vecs, 1-self.dropout)
        vecs_do2 = tf.nn.dropout(vecs, 1-self.dropout)
        vw_self = tf.matmul(vecs_do2,self.vars['order0_weights'])
        ret_self = self.act(vw_self + self.vars['order0_bias'])
        if self.bias == 'norm':
            mean,variance = tf.nn.moments(ret_self,axes=[1],keep_dims=True)
            ret_self = tf.nn.batch_normalization(ret_self,mean,variance,self.vars['order0_offset'],self.vars['order0_scale'],1e-9)
        if self.order == 0:
            return ret_self
        
        # the aggr below only applies to order 1 layers

        ret_neigh_l_subg = list()
        ret_neigh_l_fullg = list()
        offset = 0
        vw_neigh = list()
        vw_neigh_att = list()
        vw_self_att = list()
        for i in range(self.mulhead):
            vw_neigh.append(tf.matmul(vecs_do1,self.vars['order1_weights_h{}'.format(i)]))
            vw_neigh_att.append(tf.reduce_sum(vw_neigh[i]*self.vars['attention_1_h{}'.format(i)],axis=-1)\
                            + self.vars['att_bias_1_h{}'.format(i)])
            vw_self_att.append(tf.reduce_sum(vw_neigh[i]*self.vars['attention_0_h{}'.format(i)],axis=-1)\
                            + self.vars['att_bias_0_h{}'.format(i)])
        
        for i in range(self.mulhead):
            adj_weighted = self._F_edge_weight(adj_norm,vw_neigh_att[i],vw_self_att[i],offset=0)
            ret_neigh_i = self.act(tf.sparse_tensor_dense_matmul(adj_weighted,vw_neigh[i])) \
                            + self.vars['order1_bias_h{}'.format(i)]
            ret_neigh_l_subg.append(ret_neigh_i)

    
        for _adj in adj_partition_list:
            ret_neigh_la = list()
            for i in range(self.mulhead):
                adj_weighted = self._F_edge_weight(_adj,vw_neigh_att[i],vw_self_att[i],offset=offset)
                ret_neigh_i = self.act(tf.sparse_tensor_dense_matmul(adj_weighted,vw_neigh[i]) \
                                + self.vars['order1_bias_h{}'.format(i)])
                ret_neigh_la.append(ret_neigh_i)
            ret_neigh_l_fullg.append(tf.concat(ret_neigh_la,axis=1))
            offset += dim0_adj_sub
        ret_neigh = tf.cond(self.is_train, lambda: tf.concat(ret_neigh_l_subg,axis=1), lambda: tf.concat(ret_neigh_l_fullg,axis=0))
        if self.bias == 'norm':
            mean,variance = tf.nn.moments(ret_neigh,axes=[1],keep_dims=True)
            ret_neigh = tf.nn.batch_normalization(ret_neigh,mean,variance,self.vars['order1_offset'],self.vars['order1_scale'],1e-9)
        if self.aggr == 'mean':
            ret = ret_neigh + ret_self
        elif self.aggr == 'concat':
            ret = tf.concat([ret_self,ret_neigh],axis=1)
        else:
            raise NotImplementedError
        return ret

