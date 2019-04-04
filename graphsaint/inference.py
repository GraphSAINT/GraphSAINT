import pickle
import tensorflow as tf
from graphsaint.utils import *
from graphsaint.supervised_train import evaluate_full_batch,construct_placeholders,FLAGS
from graphsaint.minibatch import NodeMinibatchIterator
from graphsaint.supervised_models import Supervisedgraphsaint
from zython.logf.printf import printf
import yaml
import numpy as np
from graphsaint.metric import *


# flags to run:
#       --data_prefix   <./data/ppi>
#       --model         <./model/*.chkpt>
#       --train_config  <./train_config/*.yml>
#       --test_config   <./model/*.yml>

def evaluate_batched(sess,model,minibatch,num_nodes):
    num_cls = minibatch.class_arr.shape[-1]
    label=np.zeros((num_nodes,num_cls))
    preds=np.zeros((num_nodes,num_cls))
    loss=0
    while not minibatch.end_partial_test():
        feed_dict,label_curr=minibatch.minibatch_train_feed_dict(0,is_partial_test=True)
        nodes_id_curr=minibatch.last_batch_nodes()
        # import pdb; pdb.set_trace()
        preds_curr,loss_curr=sess.run([model.preds,model.loss],feed_dict=feed_dict)
        label[nodes_id_curr]=label_curr
        preds[nodes_id_curr]=preds_curr
        loss+=loss_curr
    f1_scores = calc_f1(label[minibatch.node_test],preds[minibatch.node_test],model.sigmoid_loss)
    return loss,f1_scores[0],f1_scores[1]

def inference_main(argv=None):
    train_params,train_phases,train_data,dims_gcn = parse_n_prepare(FLAGS)
    adj_full,adj_train,feats,class_arr,role = train_data
    adj_full_norm = adj_norm(adj_full,train_params['norm_adj'])
    num_classes = class_arr.shape[1]

    dims = dims_gcn[:-1]
    loss_type = dims_gcn[-1]

    placeholders = construct_placeholders(num_classes)
    minibatch = NodeMinibatchIterator(adj_full, adj_full_norm, adj_train, role, class_arr, placeholders, train_params)
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))#device_count={"CPU":2},inter_op_parallelism_threads=44,intra_op_parallelism_threads=44))

    model = Supervisedgraphsaint(num_classes, placeholders,
                feats, dims, train_params, loss_type, adj_full_norm, logging=True)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(var_list=tf.global_variables())
    saver.restore(sess,FLAGS.model)
    # import pdb; pdb.set_trace()

    # print("----------------------")
    # print("Full Batch Test Result")
    # print("----------------------")
    # many_run_timeline=[]
    # loss, f1_mic, f1_mac, duration = evaluate_full_batch(sess,model,minibatch,many_run_timeline)
    # printf("Full validation stats: \n\tloss={:.5f}\tf1_micro={:.5f}\tf1_macro={:.5f}",loss,f1_mic,f1_mac)
    # loss, f1_mic, f1_mac, duration = evaluate_full_batch(sess,model,minibatch,many_run_timeline,is_val=False)
    # printf("Full test stats: \n\tloss={:.5f}\tf1_micro={:.5f}\tf1_macro={:.5f}",loss,f1_mic,f1_mac)

    print("-------------------")
    print("Batched Test Result")
    print("-------------------")
    test_config=yaml.load(open(FLAGS.test_config,'r'))
    minibatch=NodeMinibatchIterator(adj_full,adj_full_norm,adj_train,role,class_arr,placeholders,train_params)
    minibatch.set_sampler(test_config,False)
    loss,f1_mic,f1_mac=evaluate_batched(sess,model,minibatch,adj_full.indptr.shape[0])
    printf("Batched test stats: \n\tloss={:.5f}\tf1_micro={:.5f}\tf1_macro={:.5f}",loss,f1_mic,f1_mac)

if __name__ == '__main__':
    tf.app.run(main=inference_main)
