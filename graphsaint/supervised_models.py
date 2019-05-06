import tensorflow as tf
from collections import namedtuple
from graphsaint.globals import *
from graphsaint.inits import *
import graphsaint.layers as layers
import graphsaint.utils as utils
import pdb


class Supervisedgraphsaint:

    def __init__(self, num_classes, placeholders, features,
            dims, train_params, type_loss, adj_full_norm, model_pretrain=None, **kwargs):
        '''
        Args:
            - placeholders: TensorFlow placeholder object.
            - features: Numpy array with node features.
            - adj: Numpy array with adjacency lists (padded with random re-samples)
            - degrees: Numpy array with node degrees.
            - sigmoid_loss: Set to true if nodes can belong to multiple classes
            - model_pretrain: contains pre-trained weights, if you are doing inferencing
        '''
        if train_params['model'] == 'gs_mean':
            self.aggregator_cls = layers.MeanAggregator
        elif train_params['model'] == 'gsaint':
            self.aggregator_cls = layers.HighOrderAggregator
        self.gcn_model = train_params['model']
        self.lr = train_params['lr']
        self.node_subgraph = placeholders['node_subgraph']
        self.nnz = placeholders['nnz']
        self.num_layers = len(dims)
        self.weight_decay = train_params['weight_decay']
        self.adj_subgraph  = placeholders['adj_subgraph']
        self.adj_subgraph_0=placeholders['adj_subgraph_0']
        self.adj_subgraph_1=placeholders['adj_subgraph_1']
        self.adj_subgraph_2=placeholders['adj_subgraph_2']
        self.adj_subgraph_3=placeholders['adj_subgraph_3']
        self.batch_norm = train_params['batch_norm']
        self.skip = train_params['skip']
        self.features = tf.Variable(tf.constant(features, dtype=DTYPE), trainable=False)
        _indices = np.column_stack(adj_full_norm.nonzero())
        _data = adj_full_norm.data
        _shape = adj_full_norm.shape
        with tf.device('/cpu:0'):
            self.adj_full_norm = tf.SparseTensorValue(_indices,_data,_shape)
        self.num_classes = num_classes
        self.sigmoid_loss = (type_loss=='sigmoid')
        _dims,_order,_act,_bias,_norm,_aggr = utils.parse_layer_yml(dims)
        self.order_layer = _order
        self.act_layer = _act
        self.bias_layer = _bias
        self.norm_layer = _norm
        self.aggr_layer = _aggr
        self.set_dims([features.shape[1]]+_dims)
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.reset_optimizer_op = tf.variables_initializer(self.optimizer.variables())
        self.loss = 0
        self.opt_op = None
        self.norm_loss = placeholders['norm_loss']
        self.is_train = placeholders['is_train']

        self.build(model_pretrain=model_pretrain)

    def set_dims(self,dims):
        if self.gcn_model == 'gs_mean':
            self.dims_feat = [dims[0]] + [2*d for d in dims[1:]]
        elif self.gcn_model == 'gsaint':
            self.dims_feat = [dims[0]] + [((self.aggr_layer[l]=='concat')*self.order_layer[l]+1)*dims[l+1] for l in range(len(dims)-1)]
        else:
            raise NotImplementedError
        self.dims_weight = [(self.dims_feat[l],dims[l+1]) for l in range(len(dims)-1)]


    def build(self, model_pretrain=None):
        """
        Build the sample graph with adj info in self.sample()
        directly feed the sampled support vectors to tf placeholder
        """
        model_pretrain_aggr = model_pretrain['meanaggr'] if model_pretrain else None
        model_pretrain_dense = model_pretrain['dense'] if model_pretrain else None
        self.aggregators = self.get_aggregators(model_pretrain=model_pretrain_aggr)
        self.outputs = self.aggregate_subgraph()
        ################
        # OUPTUT LAYER #
        ################
        self.outputs = tf.nn.l2_normalize(self.outputs, 1)
        self.node_pred = layers.Dense(self.dims_feat[-1], self.num_classes, self.weight_decay,
                dropout=self.placeholders['dropout'], act=lambda x:x, model_pretrain=model_pretrain_dense)
        self.node_preds = self.node_pred(self.outputs)

        #############
        # BACK PROP #
        #############
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
        for var in self.node_pred.vars.values():
            self.loss += self.weight_decay * tf.nn.l2_loss(var)

        # classification loss
        f_loss = tf.nn.sigmoid_cross_entropy_with_logits if self.sigmoid_loss\
                                else tf.nn.softmax_cross_entropy_with_logits
        # weighted loss due to bias in appearance of vertices
        self.loss_terms = f_loss(logits=self.node_preds,labels=self.placeholders['labels'])
        if len(self.loss_terms.shape) == 1:
            self.loss_terms = tf.reshape(self.loss_terms,(-1,1))
        self._weight_loss_batch = tf.nn.embedding_lookup(self.norm_loss, self.node_subgraph)
        self._weight_loss_batch /= tf.reduce_sum(self._weight_loss_batch)
        _loss_terms_weight = tf.linalg.matmul(tf.transpose(self.loss_terms),\
                    tf.reshape(self._weight_loss_batch,(-1,1)))
        self.loss += tf.reduce_mean(_loss_terms_weight)
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
                    bias=self.bias_layer[layer],act=self.act_layer[layer],order=self.order_layer[layer],\
                    norm=self.norm_layer[layer],aggr=self.aggr_layer[layer],is_train=self.is_train,batch_norm=self.batch_norm)
            aggregators.append(aggregator)
        return aggregators


    def aggregate_subgraph(self, batch_size=None, name=None, mode='train'):
        if mode == 'train':
            hidden = tf.nn.embedding_lookup(self.features, self.node_subgraph)
            adj = self.adj_subgraph
        else:
            hidden = self.features
            adj = self.adj_full_norm
        skip_from=-1
        skip_to=-1
        if self.skip!='noskip':
            skip_from=int(self.skip.split('-')[0])
            skip_to=int(self.skip.split('-')[1])
        for layer in range(self.num_layers):
            if layer==skip_to:
                hidden=hidden+hidden_save
            hidden = self.aggregators[layer]((hidden,adj,self.nnz,self.dims_feat[layer],self.adj_subgraph_0,self.adj_subgraph_1,self.adj_subgraph_2,self.adj_subgraph_3))
            if layer==skip_from:
                hidden_save=hidden
        return hidden

    def aggregate_fullgraph(self):
        pass
            
