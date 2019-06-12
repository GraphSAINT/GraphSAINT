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

def sparse_tensor_dense_matmul_cpu(a,b):
    # with tf.device('/cpu:0'):
    #     c=tf.sparse_tensor_dense_matmul(a,b)
    
    # partition the matrix to run in GPU
    NUM_SPLIT=4
    a_part=tf.sparse.split(sp_input=a,num_split=NUM_SPLIT,axis=0)
    c_part=[]
    for i in range(NUM_SPLIT):
        c_part.append(tf.sparse_tensor_dense_matmul(a_part[i],b))
    c=tf.concat([c_part[0],c_part[1]],0)
    for i in range(2,NUM_SPLIT):
        c=tf.concat([c,c_part[i]],0)
    return c

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
        allowed_kwargs = {'name', 'logging'}
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
        self.act = act
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



class MeanAggregator(Layer):
    """
    Aggregates via mean followed by matmul and non-linearity.
    """

    def __init__(self, dim_in, dim_out,
            dropout=0., bias=True, act=tf.nn.relu, order=1, norm=True, model_pretrain=None, aggr='concat', is_train=True, batch_norm='tf.nn', **kwargs):
        """
        model_pretrain is not None if you want to load the trained model
        model_pretrain[0] is neigh_weights
        model_pretrain[1] is self_weights
        """
        super(MeanAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        assert order == 1
        self.order = order
        self.norm = norm
        self.is_train=is_train
        self.norm=norm

        with tf.variable_scope(self.name + '_vars'):
            if model_pretrain is None:
                self.vars['neigh_weights'] = glorot([dim_in,dim_out], name='neigh_weights')
                self.vars['self_weights'] = glorot([dim_in,dim_out], name='self_weights')
            else:
                self.vars['neigh_weights'] = trained(model_pretrain[0], name='neigh_weights')
                self.vars['self_weights'] = trained(model_pretrain[1], name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([2*dim_out], name='bias')     # TODO: 2 due to concat of self and neigh vecs
            if self.norm:
                self.vars['offset'] = zeros([1,2*dim_out],name='offset')
                self.vars['scale'] = ones([1,2*dim_out],name='scale')

        if self.logging:
            self._log_vars()

        self.dim_in = dim_in
        self.dim_out = dim_out

    def _call(self, inputs):
        vecs,adj_norm,nnz,len_feat = inputs
        # TODO: introduce input dropout as a separate param of dropout
        vecs = tf.nn.dropout(vecs, 1-self.dropout)
        stride = 2**31//nnz     # this is to tackle the GPU limitation
        f_cond = lambda i,r: i<len_feat
        f_body = lambda i,r: (i+stride,tf.concat(\
            [r,tf.sparse_tensor_dense_matmul(adj_norm,vecs[:,i:i+stride])],axis=1))
        neigh_means = tf.while_loop(f_cond,f_body,\
            (stride,tf.sparse_tensor_dense_matmul(adj_norm,vecs[:,:stride])))[1]
        from_neighs = tf.matmul(neigh_means, self.vars['neigh_weights'])
        from_self = tf.matmul(vecs, self.vars["self_weights"])
        output = tf.concat([from_self, from_neighs], axis=1)
        if self.bias and not self.norm:
            output += self.vars['bias']
        with tf.variable_scope(self.name + '_vars'):
            if self.norm:
                if self.norm == 'tf.nn':
                    mean,variance = tf.nn.moments(output,axes=[1],keep_dims=True)
                    output = tf.nn.batch_normalization(output,mean,variance,self.vars['offset'],self.vars['scale'],1e-9)
                elif self.norm == 'tf.layers':
                    output=tf.layers.batch_normalization(output,training=self.is_train,renorm=False)
        return self.act(output)


class HighOrderAggregator(Layer):
    def __init__(self, dim_in, dim_out,
            dropout=0., bias=True, act=tf.nn.relu, order=2, norm=True, aggr='mean', model_pretrain=None, is_train=True, batch_norm='tf.nn', **kwargs):
        super(HighOrderAggregator,self).__init__(**kwargs)
        self.dropout = dropout
        self.bias = bias
        self.bias = True
        self.act = act
        self.order = order
        self.norm = norm
        self.aggr = aggr
        self.is_train = is_train
        self.batch_norm = batch_norm
        # TODO: enable the first layer to skip weight application, by, say, setting dim_in or dim_out=0
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
                if self.bias:
                    for o in range(self.order+1):
                        _k = 'order{}_bias'.format(o)
                        self.vars[_k] = zeros([dim_out],name=_k)
                if self.norm:
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
        if self.bias and not self.norm:
            vw += self.vars['order{}_bias'.format(order)]
        vw += self.vars['order{}_bias'.format(order)]
        vw = self.act(vw)
        if self.norm:
            if self.batch_norm == 'tf.nn':
                mean,variance = tf.nn.moments(vw,axes=[1],keep_dims=True)
                _off = 'order{}_offset'.format(order)
                _sca = 'order{}_scale'.format(order)
                vw = tf.nn.batch_normalization(vw,mean,variance,self.vars[_off],self.vars[_sca],1e-9)
            elif self.batch_norm == 'tf.layers':
                vw=tf.layers.batch_normalization(vw,training=self.is_train,renorm=False)
        #vw = self.act(vw)
        return vw

    def _call(self, inputs):
        vecs, adj_norm, nnz, len_feat, adj_0, adj_1, adj_2, adj_3, adj_4, adj_5, adj_6, adj_7 = inputs
        # TODO: where should you put dropout? here or at the end of this aggr??
        vecs = tf.nn.dropout(vecs, 1-self.dropout)
        vecs_hop = [tf.identity(vecs) for i in range(self.order+1)]
        for o in range(self.order):
            for a in range(o+1):
                ans1 = tf.sparse_tensor_dense_matmul(adj_norm,vecs_hop[o+1])
                ans2_0 = tf.sparse_tensor_dense_matmul(adj_0,vecs_hop[o+1])
                ans2_1 = tf.sparse_tensor_dense_matmul(adj_1,vecs_hop[o+1])
                ans2_2 = tf.sparse_tensor_dense_matmul(adj_2,vecs_hop[o+1])
                ans2_3 = tf.sparse_tensor_dense_matmul(adj_3,vecs_hop[o+1])
                ans2_4 = tf.sparse_tensor_dense_matmul(adj_4,vecs_hop[o+1])
                ans2_5 = tf.sparse_tensor_dense_matmul(adj_5,vecs_hop[o+1])
                ans2_6 = tf.sparse_tensor_dense_matmul(adj_6,vecs_hop[o+1])
                ans2_7 = tf.sparse_tensor_dense_matmul(adj_7,vecs_hop[o+1])
                ans2 = tf.concat([ans2_0,ans2_1,ans2_2,ans2_3,ans2_4,ans2_5,ans2_6,ans2_7],0)
                vecs_hop[o+1] = tf.cond(self.is_train,lambda: tf.identity(ans1),lambda: tf.identity(ans2))
        
        for o in range(self.order+1):
            vecs_hop[o] = self._F_nonlinear(vecs_hop[o],o)

        # ------------------------------------------------------------------------------
        #vecs_hop = []
        #for o in range(self.order+1):
        #    if self.dim_out > 0:
        #        vecs_hop.append(self._F_nonlinear(vecs,o))
        #    else:
        #        vecs_hop.append(vecs)
        #for o in range(self.order):
        #    for a in range(o+1):
        #        # v1
        #        # vecs_hop[o+1] = tf.sparse_tensor_dense_matmul(adj_norm,vecs_hop[o+1])

        #        # v2
        #        # _exceed_gpu_memory_space=tf.greater(tf.size(adj_norm),50000*50000)
        #        # vecs_hop[o+1]=tf.cond(_exceed_gpu_memory_space,lambda:sparse_tensor_dense_matmul_cpu(adj_norm,vecs_hop[o+1]),lambda:tf.sparse_tensor_dense_matmul(adj_norm,vecs_hop[o+1]))

        #        ans1=tf.sparse_tensor_dense_matmul(adj_norm,vecs_hop[o+1])
        #        ans2_0=tf.sparse_tensor_dense_matmul(adj_0,vecs_hop[o+1])
        #        ans2_1=tf.sparse_tensor_dense_matmul(adj_1,vecs_hop[o+1])
        #        ans2_2=tf.sparse_tensor_dense_matmul(adj_2,vecs_hop[o+1])
        #        ans2_3=tf.sparse_tensor_dense_matmul(adj_3,vecs_hop[o+1])
        #        ans2=tf.concat([ans2_0,ans2_1,ans2_2,ans2_3],0)
        #        vecs_hop[o+1]=tf.cond(self.is_train,lambda: tf.identity(ans1),lambda: tf.identity(ans2))
        """
        stride = 2**31//nnz
        f_cond = lambda i,r: i<len_feat
        f_body1 = lambda i,r: (i+stride,tf.concat(\
            [r,tf.sparse_tensor_dense_matmul(adj_norm,vecs_hop1[:,i:i+stride])],axis=1))
        vecs_hop1 = tf.while_loop(f_cond,f_body1,\
            (stride,tf.sparse_tensor_dense_matmul(adj_norm,vecs_hop1[:,:stride])))[1]
        f_body2 = lambda i,r: (i+stride,tf.concat(\
            [r,tf.sparse_tensor_dense_matmul(adj_norm,\
             tf.sparse_tensor_dense_matmul(adj_norm,vecs_hop2[:,i:i+stride]))],axis=1))
        vecs_hop2 = tf.while_loop(f_cond,f_body2,\
            (stride,tf.sparse_tensor_dense_matmul(adj_norm,\
             tf.sparse_tensor_dense_matmul(adj_norm,vecs_hop2[:,:stride]))))[1]
        """
        if self.aggr == 'mean':
            ret = vecs_hop[0]
            for o in range(len(vecs_hop)-1):
                ret += vecs_hop[o+1]
        elif self.aggr == 'concat':
            ret = tf.concat(vecs_hop,axis=1)
        else:
            raise NotImplementedError
        return ret
