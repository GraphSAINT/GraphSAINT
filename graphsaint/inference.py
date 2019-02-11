import pickle
import tensorflow as tf
from graphsaint.utils import *
from graphsaint.supervised_train import evaluate_full_batch,construct_placeholders,FLAGS
from graphsaint.minibatch import NodeMinibatchIterator
from graphsaint.supervised_models import Supervisedgraphsaint
from zython.logf.printf import printf


# flags to run:
#       --data_prefix   <./data/ppi>
#       --model         <./model/*.pkl>
#       --train_config  <./train_config/*.yaml>



def load_pretrained_model(name,dims):
    with open(name,'rb') as f:
        weights = pickle.load(f)
    ret = {'dense': [weights['dense_weight'],weights['dense_bias']],
           'meanaggr':[]}
    for l in range(len(dims)):
        ret['meanaggr'].append([weights['neigh_weight_{}'.format(l)],weights['self_weight_{}'.format(l)]])
    return ret


def inference_main(argv=None):
    train_params,train_phases,train_data,dims_gcn = parse_n_prepare(FLAGS)
    adj_full,adj_train,feats,class_arr,role = train_data
    num_classes = class_arr.shape[1]

    dims = dims_gcn[:-1]
    loss_type = dims_gcn[-1]

    placeholders = construct_placeholders(num_classes)
    minibatch = NodeMinibatchIterator(adj_full, adj_train, role, class_arr, placeholders)
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))#device_count={"CPU":2},inter_op_parallelism_threads=44,intra_op_parallelism_threads=44))

    pretrained_mpdel = load_pretrained_model(FLAGS.model,dims)
    model = Supervisedgraphsaint(num_classes,placeholders,feats,dims,train_params,\
                loss=loss_type,model_pretrain=pretrained_mpdel,logging=True)

    sess.run(tf.global_variables_initializer())

    loss, f1_mic, f1_mac, duration = evaluate_full_batch(sess,model,minibatch)
    printf("Full validation stats: \n\tloss={:.5f}\tf1_micro={:.5f}\tf1_macro={:.5f}",loss,f1_mic,f1_mac)
    loss, f1_mic, f1_mac, duration = evaluate_full_batch(sess,model,minibatch,is_val=False)
    printf("Full validation stats: \n\tloss={:.5f}\tf1_micro={:.5f}\tf1_macro={:.5f}",loss,f1_mic,f1_mac)
 

if __name__ == '__main__':
    tf.app.run(main=inference_main)
