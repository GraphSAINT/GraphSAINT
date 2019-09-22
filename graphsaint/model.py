import tensorflow as tf
from collections import namedtuple
from graphsaint.globals import *
from graphsaint.inits import *
import graphsaint.layers as layers
from graphsaint.utils import *
import pdb


class GraphSAINT:

    def __init__(self, num_classes, placeholders, features,
            arch_gcn, train_params, adj_full_norm, model_pretrain=None, **kwargs):
        '''
        Args:
            - placeholders: TensorFlow placeholder object.
            - features: Numpy array with node features.
            - adj: Numpy array with adjacency lists (padded with random re-samples)
            - degrees: Numpy array with node degrees.
            - sigmoid_loss: Set to true if nodes can belong to multiple classes
            - model_pretrain: contains pre-trained weights, if you are doing inferencing
        '''
        if "attention" in arch_gcn:
            self.aggregator_cls = layers.AttentionAggregator
            self.mulhead = int(arch_gcn['attention'])
        else:
            self.aggregator_cls = layers.HighOrderAggregator
            self.mulhead = 1
        self.lr = train_params['lr']
        self.node_subgraph = placeholders['node_subgraph']
        self.num_layers = len(arch_gcn['arch'].split('-'))
        self.weight_decay = train_params['weight_decay']
        self.jk = None if 'jk' not in arch_gcn else arch_gcn['jk']
        self.arch_gcn = arch_gcn
        self.adj_subgraph = placeholders['adj_subgraph']
        self.adj_subgraph_0=placeholders['adj_subgraph_0']
        self.adj_subgraph_1=placeholders['adj_subgraph_1']
        self.adj_subgraph_2=placeholders['adj_subgraph_2']
        self.adj_subgraph_3=placeholders['adj_subgraph_3']
        self.adj_subgraph_4=placeholders['adj_subgraph_4']
        self.adj_subgraph_5=placeholders['adj_subgraph_5']
        self.adj_subgraph_6=placeholders['adj_subgraph_6']
        self.adj_subgraph_7=placeholders['adj_subgraph_7']
        self.features = tf.Variable(tf.constant(features, dtype=DTYPE), trainable=False)
        self.dualGPU=FLAGS.dualGPU
        _indices = np.column_stack(adj_full_norm.nonzero())
        _data = adj_full_norm.data
        _shape = adj_full_norm.shape
        with tf.device('/cpu:0'):
            self.adj_full_norm = tf.SparseTensorValue(_indices,_data,_shape)
        self.num_classes = num_classes
        self.sigmoid_loss = (arch_gcn['loss']=='sigmoid')
        _dims,self.order_layer,self.act_layer,self.bias_layer,self.aggr_layer = parse_layer_yml(arch_gcn,features.shape[1])
        # get layer index for each conv layer, useful for jk net last layer aggregation
        self.set_idx_conv()
        self.set_dims(_dims)
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)

        self.loss = 0
        self.opt_op = None
        self.norm_loss = placeholders['norm_loss']
        self.is_train = placeholders['is_train']
        
        self.debug = list()

        self.build(model_pretrain=model_pretrain)

    def set_dims(self,dims):
        self.dims_feat = [dims[0]] + [((self.aggr_layer[l]=='concat')*self.order_layer[l]+1)*dims[l+1] for l in range(len(dims)-1)]
        self.dims_weight = [(self.dims_feat[l],dims[l+1]) for l in range(len(dims)-1)]

    def set_idx_conv(self):
        idx_conv = np.where(np.array(self.order_layer)>=1)[0]
        idx_conv = list(idx_conv[1:] - 1)
        idx_conv.append(len(self.order_layer)-1)
        self.idx_conv = idx_conv


    def build(self, model_pretrain=None):
        """
        Build the sample graph with adj info in self.sample()
        directly feed the sampled support vectors to tf placeholder
        """
        model_pretrain_aggr = model_pretrain['meanaggr'] if model_pretrain else None
        model_pretrain_dense = model_pretrain['dense'] if model_pretrain else None
        self.aggregators = self.get_aggregators(model_pretrain=model_pretrain_aggr)
        _outputs_l = self.aggregate_subgraph()
        if self.jk == 'concat':
            _dim_input_jk = np.array(self.dims_feat)[np.array(self.idx_conv)+1].sum()
        else:
            _dim_input_jk = self.dims_feat[-1]
        self.layer_jk = layers.JumpingKnowledge(self.arch_gcn,_dim_input_jk,mode=self.jk)
        self.outputs = self.layer_jk([_outputs_l, self.idx_conv])
        # OUPTUT LAYER
        self.outputs = tf.nn.l2_normalize(self.outputs, 1)
        _dim_final = self.arch_gcn['dim'] if self.jk else self.dims_feat[-1]
        self.layer_pred = layers.Dense(_dim_final, self.num_classes, self.weight_decay,
                dropout=self.placeholders['dropout'], act='I', model_pretrain=model_pretrain_dense)
        self.node_preds = self.layer_pred(self.outputs)

        # BACK PROP
        self._loss()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            grads_and_vars = self.optimizer.compute_gradients(self.loss)
            clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
                    for grad, var in grads_and_vars]
            self.grad, _ = clipped_grads_and_vars[0]
            self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)
        self.preds = self.predict()


    def _loss(self):
        # these are all the trainable var
        for aggregator in self.aggregators:
            for var in aggregator.vars.values():
                self.loss += self.weight_decay * tf.nn.l2_loss(var)
        for var in self.layer_pred.vars.values():
            self.loss += self.weight_decay * tf.nn.l2_loss(var)

        # classification loss
        f_loss = tf.nn.sigmoid_cross_entropy_with_logits if self.sigmoid_loss\
                                else tf.nn.softmax_cross_entropy_with_logits
        # weighted loss due to bias in appearance of vertices
        self.loss_terms = f_loss(logits=self.node_preds,labels=self.placeholders['labels'])
        if len(self.loss_terms.shape) == 1:
            self.loss_terms = tf.reshape(self.loss_terms,(-1,1))
        self._weight_loss_batch = tf.nn.embedding_lookup(self.norm_loss, self.node_subgraph)
        _loss_terms_weight = tf.linalg.matmul(tf.transpose(self.loss_terms),\
                    tf.reshape(self._weight_loss_batch,(-1,1)))
        self.loss += tf.reduce_sum(_loss_terms_weight)
        tf.summary.scalar('loss', self.loss)

    def predict(self):
        return tf.nn.sigmoid(self.node_preds) if self.sigmoid_loss \
                else tf.nn.softmax(self.node_preds)


    def get_aggregators(self,name=None,model_pretrain=None):
        aggregators = []
        if model_pretrain is None:
            model_pretrain = [None]*self.num_layers
        for layer in range(self.num_layers):
            aggregator = self.aggregator_cls(self.dims_weight[layer][0], self.dims_weight[layer][1],
                    dropout=self.placeholders['dropout'],name=name,model_pretrain=model_pretrain[layer],
                    act=self.act_layer[layer],order=self.order_layer[layer],aggr=self.aggr_layer[layer],\
                    is_train=self.is_train,bias=self.bias_layer[layer],logging=FLAGS.logging,\
                    I_vector=self.placeholders['I_vector'],mulhead=self.mulhead)
            aggregators.append(aggregator)
        return aggregators


    def aggregate_subgraph(self, batch_size=None, name=None, mode='train'):
        if mode == 'train':
            hidden = tf.nn.embedding_lookup(self.features, self.node_subgraph)
            adj = self.adj_subgraph
        else:
            hidden = self.features
            adj = self.adj_full_norm
        ret_l = list()
        if not FLAGS.dualGPU:
            for layer in range(self.num_layers):
                hidden,_ = self.aggregators[layer]((hidden,adj,self.dims_feat[layer],self.adj_subgraph_0,self.adj_subgraph_1,self.adj_subgraph_2,\
                        self.adj_subgraph_3,self.adj_subgraph_4,self.adj_subgraph_5,self.adj_subgraph_6,self.adj_subgraph_7))
                ret_l.append(hidden)
                self.debug.append(_)
        else:
            split=int(self.num_layers/2)
            with tf.device('/gpu:0'):
                for layer in range(split):
                    hidden = self.aggregators[layer]((hidden,adj,self.dims_feat[layer],self.adj_subgraph_0,self.adj_subgraph_1,self.adj_subgraph_2,\
                        self.adj_subgraph_3,self.adj_subgraph_4,self.adj_subgraph_5,self.adj_subgraph_6,self.adj_subgraph_7))
                    ret_l.append(hidden)
            with tf.device('/gpu:1'):
                for layer in range(split,self.num_layers):
                    hidden = self.aggregators[layer]((hidden,adj,self.dims_feat[layer],self.adj_subgraph_0,self.adj_subgraph_1,self.adj_subgraph_2,\
                        self.adj_subgraph_3,self.adj_subgraph_4,self.adj_subgraph_5,self.adj_subgraph_6,self.adj_subgraph_7))
                    ret_l.append(hidden)
        return ret_l

