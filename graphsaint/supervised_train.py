from graphsaint.globals import *
from graphsaint.inits import *
from graphsaint.supervised_models import Supervisedgraphsaint
from graphsaint.minibatch import NodeMinibatchIterator
from graphsaint.utils import *
from graphsaint.metric import *
from tensorflow.python.client import timeline

import sys, os, random
import yaml
import cProfile
import pickle
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
from zython.logf.printf import printf
import time
import datetime
import pdb
import getpass
import json

class TimeLiner:
    _timeline_dict = None

    def update_timeline(self, chrome_trace):
        # convert crome trace to python dict
        chrome_trace_dict = json.loads(chrome_trace)
        # for first run store full trace
        if self._timeline_dict is None:
            self._timeline_dict = chrome_trace_dict
        # for other - update only time consumption, not definitions
        else:
            for event in chrome_trace_dict['traceEvents']:
                # events time consumption started with 'ts' prefix
                if 'ts' in event:
                    self._timeline_dict['traceEvents'].append(event)

    def save(self, f_name):
        with open(f_name, 'w') as f:
            json.dump(self._timeline_dict, f)

def evaluate_full_batch(sess,model,minibatch_iter,many_runs_timeline,is_val=True,is_valtest=False):
    """
    Full batch evaluation
    """
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    t1 = time.time()
    num_cls = minibatch_iter.class_arr.shape[-1]
    feed_dict, labels = minibatch_iter.minibatch_train_feed_dict(0.,is_val=True,is_test=True)
    if FLAGS.timeline:
        preds,loss = sess.run([model.preds, model.loss], feed_dict=feed_dict, options=options, run_metadata=run_metadata)
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        many_runs_timeline.append(chrome_trace)
    else:
        preds,loss = sess.run([model.preds, model.loss], feed_dict=feed_dict)
    if is_valtest:
        node_val_test = np.concatenate((minibatch_iter.node_val,minibatch_iter.node_test))
    else:
        node_val_test = minibatch_iter.node_val if is_val else minibatch_iter.node_test
    t2 = time.time()
    f1_scores = calc_f1(labels[node_val_test],preds[node_val_test],model.sigmoid_loss)
    return loss, f1_scores[0], f1_scores[1], (t2-t1)



def construct_placeholders(num_classes):
    placeholders = {
        'labels': tf.placeholder(DTYPE, shape=(None, num_classes), name='labels'),
        'node_subgraph': tf.placeholder(tf.int32, shape=(None), name='node_subgraph'),
        'nnz': tf.placeholder(tf.int32, shape=(None), name='adj_nnz'),
        'dropout': tf.placeholder(DTYPE, shape=(None), name='dropout'),
        'adj_subgraph' : tf.sparse_placeholder(DTYPE,name='adj_subgraph'),
        'adj_subgraph_0' : tf.sparse_placeholder(DTYPE,name='adj_subgraph_0'),
        'adj_subgraph_1' : tf.sparse_placeholder(DTYPE,name='adj_subgraph_1'),
        'adj_subgraph_2' : tf.sparse_placeholder(DTYPE,name='adj_subgraph_2'),
        'adj_subgraph_3' : tf.sparse_placeholder(DTYPE,name='adj_subgraph_3'),
        'norm_weight': tf.placeholder(DTYPE,shape=(None),name='loss_weight'),
        'is_train': tf.placeholder(tf.bool, shape=(None), name='is_train')
    }
    return placeholders


pr = cProfile.Profile()



#########
# TRAIN #
#########
def prepare(train_data,train_params,dims_gcn):
    adj_full,adj_train,feats,class_arr,role = train_data
    adj_full = adj_full.astype(np.int32)
    adj_train = adj_train.astype(np.int32)
    adj_full_norm = adj_norm(adj_full,train_params['norm_adj'])
    num_classes = class_arr.shape[1]

    dims = dims_gcn[:-1]
    loss_type = dims_gcn[-1]

    placeholders = construct_placeholders(num_classes)
    minibatch = NodeMinibatchIterator(adj_full, adj_full_norm, adj_train, role, class_arr, placeholders, train_params)
    model = Supervisedgraphsaint(num_classes, placeholders,
                feats, dims, train_params, loss_type, adj_full_norm, logging=True)

    # config.gpu_options.allow_growth = True
    # config.allow_soft_placement = True

    # Initialize session
    sess = tf.Session(config=tf.ConfigProto(device_count={"CPU":40},inter_op_parallelism_threads=44,intra_op_parallelism_threads=44,log_device_placement=FLAGS.log_device_placement))
    # sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))
    ph_misc_stat = {'val_f1_micro': tf.placeholder(DTYPE, shape=()),
                    'val_f1_macro': tf.placeholder(DTYPE, shape=()),
                    'train_f1_micro': tf.placeholder(DTYPE, shape=()),
                    'train_f1_macro': tf.placeholder(DTYPE, shape=()),
                    'time_per_batch': tf.placeholder(DTYPE, shape=()),
                    'time_per_epoch': tf.placeholder(DTYPE, shape=()),
                    'size_subgraph': tf.placeholder(tf.int32, shape=()),
	            'learning_rate': tf.placeholder(DTYPE,shape=()),
                    'epoch_sample_time': tf.placeholder(DTYPE,shape=())}
    merged = tf.summary.merge_all()

    with tf.name_scope('summary'):
        _misc_val_f1_micro = tf.summary.scalar('val_f1_micro', ph_misc_stat['val_f1_micro'])
        _misc_val_f1_macro = tf.summary.scalar('val_f1_macro', ph_misc_stat['val_f1_macro'])
        _misc_train_f1_micro = tf.summary.scalar('train_f1_micro', ph_misc_stat['train_f1_micro'])
        _misc_train_f1_macro = tf.summary.scalar('train_f1_macro', ph_misc_stat['train_f1_macro'])
        _misc_time_per_batch = tf.summary.scalar('time_per_batch',ph_misc_stat['time_per_batch'])
        _misc_time_per_epoch = tf.summary.scalar('time_per_epoch',ph_misc_stat['time_per_epoch'])
        _misc_size_subgraph = tf.summary.scalar('size_subgraph',ph_misc_stat['size_subgraph'])
        _misc_learning_rate = tf.summary.scalar('learning_rate',ph_misc_stat['learning_rate'])
        _misc_sample_time = tf.summary.scalar('epoch_sample_time',ph_misc_stat['epoch_sample_time'])

    misc_stats = tf.summary.merge([_misc_val_f1_micro,_misc_val_f1_macro,_misc_train_f1_micro,_misc_train_f1_macro,
                    _misc_time_per_batch,_misc_time_per_epoch,_misc_size_subgraph,_misc_learning_rate,_misc_sample_time])
    summary_writer = tf.summary.FileWriter(log_dir(train_params,dims,FLAGS.train_config,FLAGS.data_prefix,git_branch,git_rev,timestamp), sess.graph)
    # Init variables
    sess.run(tf.global_variables_initializer())
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    return model,minibatch, sess, [merged,misc_stats],ph_misc_stat, summary_writer



def train(train_phases,train_params,dims_gcn,model,minibatch,\
            sess,train_stat,ph_misc_stat,summary_writer):
    import time
    avg_time = 0.0
    timing_steps = 0

    # ----------------------- tf saver
    saver = tf.train.Saver(var_list=tf.global_variables())

    epoch_ph_start = 0
    f1mic_best = 0
    e_best = 0
    time_calc_f1 = 0
    time_qest = 0
    time_train = 0
    time_prepare = 0
    timestamp_chkpt = time.time()
    model_rand_serial = random.randint(1,1000)
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    # many_runs_timeline = TimeLiner()
    many_runs_timeline=[]
    for ip,phase in enumerate(train_phases):
        tset_start = time.time()
        minibatch.set_sampler(phase,train_params['norm_weight'])
        tset_end = time.time()
        time_qest += tset_end-tset_start
        sess.run(model.reset_optimizer_op)
        num_batches = minibatch.num_training_batches()
        printf('START PHASE {:4d}',ip)
        for e in range(epoch_ph_start,int(phase['end'])):
            printf('Epoch {:4d}',e)
            minibatch.shuffle()
            l_loss_tr = list()
            l_f1mic_tr = list()
            l_f1mac_tr = list()
            l_size_subg = list()
            time_train_ep = 0
            time_prepare_ep = 0
            time_timeline_ep=0
            time_mask = 0
            time_list = 0
            while not minibatch.end():
                t0 = time.time()
                feed_dict, labels = minibatch.minibatch_train_feed_dict(phase['dropout'],is_val=False,is_test=False)
                t1 = time.time()
                if FLAGS.timeline:
                    _,__,loss_train,pred_train = sess.run([train_stat[0], \
                            model.opt_op, model.loss, model.preds], feed_dict=feed_dict,
                            options=options, run_metadata=run_metadata)
                    t2 = time.time()
                    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                    chrome_trace = fetched_timeline.generate_chrome_trace_format()
                    many_runs_timeline.append(chrome_trace)
                    t3=time.time()
                    time_timeline_ep+=t3-t2
                else:
                    _,__,loss_train,pred_train = sess.run([train_stat[0], \
                            model.opt_op, model.loss, model.preds], feed_dict=feed_dict)
                    t2 = time.time()
                time_train_ep += t2-t1
                time_prepare_ep += t1-t0
                if not minibatch.batch_num % FLAGS.print_every:
                    t3 = time.time()
                    f1_mic,f1_mac = calc_f1(labels,pred_train,dims_gcn[-1])
                    printf("Iter {:4d}\ttrain loss {:.5f}\tmic {:5f}\tmac {:5f}",\
                        minibatch.batch_num,loss_train,f1_mic,f1_mac,type=None)
                    l_loss_tr.append(loss_train)
                    l_f1mic_tr.append(f1_mic)
                    l_f1mac_tr.append(f1_mac)
                    l_size_subg.append(minibatch.size_subgraph)
                    t4 = time.time()
                    time_calc_f1 += t4 - t3
            if FLAGS.timeline:
                print('train time: {:4.2f}\tprepare time: {:4.2f}\ttimeline time: {:4.2f}'.format(time_train_ep,time_prepare_ep,time_timeline_ep)) 
            else:
                print('train time: {:4.2f}\tprepare time: {:4.2f}'.format(time_train_ep,time_prepare_ep))
            time_train += time_train_ep
            time_prepare += time_prepare_ep
            if e % 1 == 0:
                loss_val,f1mic_val,f1mac_val,time_eval = \
                        evaluate_full_batch(sess,model,minibatch,many_runs_timeline,is_val=True)
                if f1mic_val > f1mic_best:
                    f1mic_best = f1mic_val
                    e_best = e
                    # ---------- try saver
                    tsave=time.time()
                    # savepath = saver.save(sess, '/raid/users/{}/models/saved_model_{}_rand{}.chkpt'.format(getpass.getuser(),timestamp_chkpt,model_rand_serial))
                    savepath = saver.save(sess, './temp_model_{}_rand{}.chkpt'.format(timestamp_chkpt,model_rand_serial))
                    print('saver time: {:4.2f}'.format(time.time()-tsave))
                printf('   val loss {:.5f}\tmic {:.5f}\tmac {:.5f}',loss_val,f1mic_val,f1mac_val)
                printf('   avg train loss {:.5f}\tmic {:.5f}\tmac {:.5f}',f_mean(l_loss_tr),f_mean(l_f1mic_tr),f_mean(l_f1mac_tr))
                
                if FLAGS.tensorboard:
                    misc_stat = sess.run([train_stat[1]],feed_dict={\
                                            ph_misc_stat['val_f1_micro']: f1mic_val,
                                            ph_misc_stat['val_f1_macro']: f1mac_val,
                                            ph_misc_stat['train_f1_micro']: f_mean(l_f1mic_tr),
                                            ph_misc_stat['train_f1_macro']: f_mean(l_f1mac_tr),
                                            ph_misc_stat['time_per_batch']: 0,#t_epoch/num_batches,
                                            ph_misc_stat['time_per_epoch']: time_train_ep+time_prepare_ep,#t_epoch,
                                            ph_misc_stat['size_subgraph']: f_mean(l_size_subg),
                                            ph_misc_stat['learning_rate']: 0,#curr_learning_rate,
                                            ph_misc_stat['epoch_sample_time']: 0})#t_epoch_sampling})
                    # tensorboard visualization
                    summary_writer.add_summary(_, e)
                    summary_writer.add_summary(misc_stat[0], e)
        epoch_ph_start = int(phase['end'])
    #saver.save(sess, 'models/{data}'.format(data=FLAGS.data_prefix.split('/')[-1]),global_step=e)
    #save_model_weights(weight_cur,FLAGS.data_prefix.split('/')[-1],e_best,FLAGS.train_config)
    #reload_model_weights(sess,model,weight_cur)
    printf("Optimization Finished!",type='WARN')
    timelines = TimeLiner()
    for tl in many_runs_timeline:
        timelines.update_timeline(tl)
    timelines.save('timeline.json')
    # ---------- try reloading
    # saver.restore(sess, '/raid/users/{}/models/saved_model_{}_rand{}.chkpt'.format(getpass.getuser(),timestamp_chkpt,model_rand_serial))
    saver.restore(sess, './temp_model_{}_rand{}.chkpt'.format(timestamp_chkpt,model_rand_serial))
    loss_val, f1mic_val, f1mac_val, duration = evaluate_full_batch(sess,model,minibatch,many_runs_timeline)
    printf("Full validation stats: \n\tloss={:.5f}\tf1_micro={:.5f}\tf1_macro={:.5f}",loss_val,f1mic_val,f1mac_val)
    loss_test, f1mic_test, f1mac_test, duration = evaluate_full_batch(sess,model,minibatch,many_runs_timeline,is_val=False)
    printf("Full test stats: \n\tloss={:.5f}\tf1_micro={:.5f}\tf1_macro={:.5f}",loss_test,f1mic_test,f1mac_test)
    return {'loss_val_opt':loss_val,'f1mic_val_opt':f1mic_val,'f1mac_val_opt':f1mac_val,\
            'loss_test_opt':loss_test,'f1mic_test_opt':f1mic_test,'f1mac_test_opt':f1mac_test,\
            'epoch_best':e_best,
            'time_train': time_train}


########
# MAIN #
########

def train_main(argv=None,**kwargs):
    train_config = None if 'train_config' not in kwargs.keys() else kwargs['train_config']
    train_params,train_phases,train_data,dims_gcn = parse_n_prepare(FLAGS,train_config=train_config)
    model,minibatch,sess,train_stat,ph_misc_stat,summary_writer = prepare(train_data,train_params,dims_gcn)
    time_start = time.time()
    ret = train(train_phases,train_params,dims_gcn,model,minibatch,sess,train_stat,ph_misc_stat,summary_writer)
    time_end = time.time()
    print('training time: ')
    print(time.strftime("%H:%M:%S",time.gmtime(time_end-time_start)))
    with open(FNAME_RET,'wb') as f:
        pickle.dump(ret,f)
    return ret


if __name__ == '__main__':
    tf.app.run(main=train_main)

