import tensorflow as tf
from graphsaint.inits import glorot,zeros,trained,ones,xavier,uniform
from graphsaint.globals import *

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
        allowed_kwargs = {'name', 'logging', 'I_vector'}
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


class Dense(Layer):
    """Dense layer."""
    # dense should not act as relu
    def __init__(self, dim_in, dim_out, weight_decay, dropout=0.,
                 act=lambda x:x, bias=True, model_pretrain=None, **kwargs):
        """
        model_pretrain is not None if you want to load the trained model
        model_pretrain[0] is weights
        model_pretrain[1] is bias
        """
        super(Dense, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = F_ACT[act]
        self.bias = bias
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.weight_decay = weight_decay

        with tf.variable_scope(self.name + '_vars'):
            if model_pretrain is None:
                self.vars['weights'] = tf.get_variable('weights', shape=(dim_in, dim_out),
                                         dtype=DTYPE,
                                         initializer=tf.contrib.layers.xavier_initializer(),
                                         regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay))
                if self.bias:
                    self.vars['bias'] = zeros([dim_out],name='bias')
            else:
                self.vars['weights'] = trained(model_pretrain[0], name='weight')
                if self.bias:
                    self.vars['bias'] = trained(model_pretrain[1], name='bias')
        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = tf.nn.dropout(inputs, 1-self.dropout)
        output = tf.matmul(x, self.vars['weights'])
        if self.bias:
            output += self.vars['bias']
        return self.act(output)


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
    def __init__(self, dim_in, dim_out,
            dropout=0., act='relu', order=1, aggr='mean', model_pretrain=None, is_train=True, bias='norm', **kwargs):
        super(HighOrderAggregator,self).__init__(**kwargs)
        self.dropout = dropout
        self.bias = bias
        self.act = F_ACT[act]
        self.order = order
        self.aggr = aggr
        self.is_train = is_train
        if dim_out > 0:
            with tf.variable_scope(self.name + '_vars'):
                if model_pretrain is None:
                    for o in range(self.order+1):
                        _k = 'order{}_weights'.format(o)
                        self.vars[_k] = glorot([dim_in,dim_out],name=_k)
                else:
                    for o in range(self.order+1):
                        _k = 'order{}_weights'.format(o)
                        self.vars[_k] = trained(model_pretrain[0], name=_k)
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
        vecs, adj_norm, len_feat, adj_0, adj_1, adj_2, adj_3, adj_4, adj_5, adj_6, adj_7 = inputs
        vecs = tf.nn.dropout(vecs, 1-self.dropout)
        vecs_hop = [tf.identity(vecs) for o in range(self.order+1)]
        for o in range(self.order):
            for a in range(o+1):
                ans1=tf.sparse_tensor_dense_matmul(adj_norm,vecs_hop[o+1])
                ans2_0=tf.sparse_tensor_dense_matmul(adj_0,vecs_hop[o+1])
                ans2_1=tf.sparse_tensor_dense_matmul(adj_1,vecs_hop[o+1])
                ans2_2=tf.sparse_tensor_dense_matmul(adj_2,vecs_hop[o+1])
                ans2_3=tf.sparse_tensor_dense_matmul(adj_3,vecs_hop[o+1])
                ans2_4=tf.sparse_tensor_dense_matmul(adj_4,vecs_hop[o+1])
                ans2_5=tf.sparse_tensor_dense_matmul(adj_5,vecs_hop[o+1])
                ans2_6=tf.sparse_tensor_dense_matmul(adj_6,vecs_hop[o+1])
                ans2_7=tf.sparse_tensor_dense_matmul(adj_7,vecs_hop[o+1])
                ans2=tf.concat([ans2_0,ans2_1,ans2_2,ans2_3,ans2_4,ans2_5,ans2_6,ans2_7],0)
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
        return ret,[adj_norm]


class AttentionAggregator(Layer):
    def __init__(self, dim_in, dim_out,
            dropout=0., act='relu', order=1, aggr='mean', model_pretrain=None, is_train=True, bias='norm', **kwargs):
        assert order == 1, "now only support attention for order 1 layers"
        super(AttentionAggregator,self).__init__(**kwargs)
        self.dropout = dropout
        self.bias = bias
        self.act = F_ACT[act]
        self.order = 1      # for attention, right now we only support order 1 (i.e., self + 1-hop neighbor)
        self.aggr = aggr
        self.is_train = is_train
        with tf.variable_scope(self.name + '_vars'):
            if model_pretrain is None:
                for o in range(self.order+1):
                    _k = 'order{}_weights'.format(o)
                    self.vars[_k] = glorot([dim_in,dim_out],name=_k)
            else:
                for o in range(self.order+1):
                    _k = 'order{}_weights'.format(o)
                    self.vars[_k] = trained(model_pretrain[0], name=_k)
            for o in range(self.order+1):
                _k = 'order{}_bias'.format(o)
                self.vars[_k] = zeros([dim_out],name=_k)
                _k = 'order{}_bias_after'.format(o)
                self.vars[_k] = zeros([dim_out],name=_k)
            if self.bias == 'norm':
                for o in range(self.order+1):
                    _k1 = 'order{}_offset'.format(o)
                    _k2 = 'order{}_scale'.format(o)
                    self.vars[_k1] = zeros([1,dim_out],name=_k1)
                    self.vars[_k2] = ones([1,dim_out],name=_k2)
            self.vars['attention_0'] = glorot([1,dim_out], name='attention_0')      # to apply to self feat     # tf.ones([1,dim_out])
            self.vars['attention_1'] = glorot([1,dim_out], name='attention_1')      # to apply to neigh feat
            self.vars['att_bias_0'] = zeros([1], name='att_bias_0')
            self.vars['att_bias_1'] = zeros([1], name='att_bias_1')
        print('>> layer {}, dim: [{},{}]'.format(self.name, dim_in, dim_out))
        if self.logging:
            self._log_vars()

        self.dim_in = dim_in
        self.dim_out = dim_out

        self.I_vector = kwargs['I_vector']

    def _call(self, inputs):
    
        vecs, adj_norm, len_feat, adj_0, adj_1, adj_2, adj_3, adj_4, adj_5, adj_6, adj_7 = inputs
        vecs = tf.nn.dropout(vecs, 1-self.dropout)
        adj_mask = tf.dtypes.cast(tf.dtypes.cast(adj_norm, tf.bool), tf.float32)

        vw_neigh = tf.matmul(vecs,self.vars['order1_weights'])
        vw_self = tf.matmul(vecs,self.vars['order0_weights'])
        #-vw_neigh += self.vars['order1_bias']
        #-vw_self += self.vars['order0_bias']
        vw_neigh_att = tf.reduce_sum(vw_neigh * self.vars['attention_1'], axis=-1)
        vw_self_att = tf.reduce_sum(vw_neigh * self.vars['attention_0'], axis=-1)       # NOTE: here we still use vw_neigh
        vw_neigh_att += self.vars['att_bias_1']
        vw_self_att += self.vars['att_bias_0']
        a1 = tf.SparseTensor(adj_mask.indices, tf.nn.embedding_lookup(vw_neigh_att, adj_mask.indices[:,1]), adj_mask.dense_shape)
        a2 = tf.SparseTensor(adj_mask.indices, tf.nn.embedding_lookup(vw_self_att, adj_mask.indices[:,0]),adj_mask.dense_shape)
        a = tf.SparseTensor(a1.indices,a1.values+a2.values,a1.dense_shape)
        a = tf.SparseTensor(a.indices,tf.nn.leaky_relu(a.values),a.dense_shape)
        ##a_exp = tf.SparseTensor(a.indices,tf.math.exp(a.values),a.dense_shape)
        #a_exp = tf.math.exp(a) * adj_mask
        #a_exp dot I
        ##a_exp_sum = tf.sparse_tensor_dense_matmul(a_exp,self.I_vector)
        deg = tf.squeeze(tf.sparse_tensor_dense_matmul(adj_mask,self.I_vector),axis=1)
        ##a_exp_sum = tf.SparseTensor(a_exp.indices,tf.nn.embedding_lookup(tf.squeeze(a_exp_sum,axis=1), a_exp.indices[:,0]), a_exp.dense_shape)
        ##alpha = tf.SparseTensor(a_exp.indices, a_exp.values/a_exp_sum.values, a_exp.dense_shape)
        alpha = tf.sparse.softmax(a)
        _n = tf.nn.embedding_lookup(deg, adj_mask.indices[:,0])
        alpha = tf.SparseTensor(alpha.indices, alpha.values*_n,alpha.dense_shape)

        adj_weighted = tf.SparseTensor(adj_norm.indices, adj_norm.values * alpha.values, adj_norm.dense_shape)
        ret_neigh = tf.sparse_tensor_dense_matmul(adj_weighted,vw_neigh)
        ret_self = vw_self
        ret_neigh += self.vars['order1_bias']
        ret_self += self.vars['order0_bias']
        ret_neigh = self.act(ret_neigh)
        ret_self = self.act(ret_self)
        if self.aggr == 'mean':
            ret = ret_neigh + ret_self
        elif self.aggr == 'concat':
            ret = tf.concat([ret_self,ret_neigh],axis=1)
        else:
            raise NotImplementedError
        return ret, [_n, alpha]

