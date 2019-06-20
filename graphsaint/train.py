from graphsaint.globals import *
from graphsaint.inits import *
from graphsaint.model import GraphSAINT
from graphsaint.minibatch import Minibatch
from graphsaint.utils import *
from graphsaint.metric import *
from tensorflow.python.client import timeline

import sys, os, random
import tensorflow as tf
import numpy as np
import time
import pdb
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

def evaluate_full_batch(sess,model,minibatch_iter,many_runs_timeline,mode):
    """
    Full batch evaluation
    NOTE: HERE GCN RUNS THROUGH THE FULL GRAPH. HOWEVER, WE CALCULATE F1 SCORE
        FOR VALIDATION / TEST NODES ONLY. 
    """
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    t1 = time.time()
    num_cls = minibatch_iter.class_arr.shape[-1]
    feed_dict, labels = minibatch_iter.feed_dict(0.,mode)
    if FLAGS.timeline:
        preds,loss = sess.run([model.preds, model.loss], feed_dict=feed_dict, options=options, run_metadata=run_metadata)
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        many_runs_timeline.append(chrome_trace)
    else:
        preds,loss = sess.run([model.preds, model.loss], feed_dict=feed_dict)
    node_val_test = minibatch_iter.node_val if mode=='val' else minibatch_iter.node_test
    t2 = time.time()
    f1_scores = calc_f1(labels[node_val_test],preds[node_val_test],model.sigmoid_loss)
    return loss, f1_scores[0], f1_scores[1], (t2-t1)



def construct_placeholders(num_classes):
    placeholders = {
        'labels': tf.placeholder(DTYPE, shape=(None, num_classes), name='labels'),
        'node_subgraph': tf.placeholder(tf.int32, shape=(None), name='node_subgraph'),
        'dropout': tf.placeholder(DTYPE, shape=(None), name='dropout'),
        'adj_subgraph' : tf.sparse_placeholder(DTYPE,name='adj_subgraph'),
        'adj_subgraph_last': tf.sparse_placeholder(DTYPE,name='adj_subgraph_last'),
        'adj_subgraph_0' : tf.sparse_placeholder(DTYPE,name='adj_subgraph_0'),
        'adj_subgraph_1' : tf.sparse_placeholder(DTYPE,name='adj_subgraph_1'),
        'adj_subgraph_2' : tf.sparse_placeholder(DTYPE,name='adj_subgraph_2'),
        'adj_subgraph_3' : tf.sparse_placeholder(DTYPE,name='adj_subgraph_3'),
        'adj_subgraph_4' : tf.sparse_placeholder(DTYPE,name='adj_subgraph_4'),
        'adj_subgraph_5' : tf.sparse_placeholder(DTYPE,name='adj_subgraph_5'),
        'adj_subgraph_6' : tf.sparse_placeholder(DTYPE,name='adj_subgraph_6'),
        'adj_subgraph_7' : tf.sparse_placeholder(DTYPE,name='adj_subgraph_7'),
        'norm_loss': tf.placeholder(DTYPE,shape=(None),name='norm_loss'),
        'is_train': tf.placeholder(tf.bool, shape=(None), name='is_train')
    }
    return placeholders





#########
# TRAIN #
#########
def prepare(train_data,train_params,arch_gcn):
    adj_full,adj_train,feats,class_arr,role = train_data
    adj_full = adj_full.astype(np.int32)
    adj_train = adj_train.astype(np.int32)
    adj_full_norm = adj_norm(adj_full)
    num_classes = class_arr.shape[1]

    placeholders = construct_placeholders(num_classes)
    minibatch = Minibatch(adj_full, adj_full_norm, adj_train, role, class_arr, placeholders, train_params)
    model = GraphSAINT(num_classes, placeholders,
                feats, arch_gcn, train_params, adj_full_norm, logging=True)

    # config.gpu_options.allow_growth = True
    # config.allow_soft_placement = True

    # Initialize session
    sess = tf.Session(config=tf.ConfigProto(device_count={"CPU":40},inter_op_parallelism_threads=44,intra_op_parallelism_threads=44,log_device_placement=FLAGS.log_device_placement))
    ph_misc_stat = {'val_f1_micro': tf.placeholder(DTYPE, shape=()),
                    'val_f1_macro': tf.placeholder(DTYPE, shape=()),
                    'train_f1_micro': tf.placeholder(DTYPE, shape=()),
                    'train_f1_macro': tf.placeholder(DTYPE, shape=()),
                    'time_per_epoch': tf.placeholder(DTYPE, shape=()),
                    'size_subgraph': tf.placeholder(tf.int32, shape=())}
    merged = tf.summary.merge_all()

    with tf.name_scope('summary'):
        _misc_val_f1_micro = tf.summary.scalar('val_f1_micro', ph_misc_stat['val_f1_micro'])
        _misc_val_f1_macro = tf.summary.scalar('val_f1_macro', ph_misc_stat['val_f1_macro'])
        _misc_train_f1_micro = tf.summary.scalar('train_f1_micro', ph_misc_stat['train_f1_micro'])
        _misc_train_f1_macro = tf.summary.scalar('train_f1_macro', ph_misc_stat['train_f1_macro'])
        _misc_time_per_epoch = tf.summary.scalar('time_per_epoch',ph_misc_stat['time_per_epoch'])
        _misc_size_subgraph = tf.summary.scalar('size_subgraph',ph_misc_stat['size_subgraph'])

    misc_stats = tf.summary.merge([_misc_val_f1_micro,_misc_val_f1_macro,_misc_train_f1_micro,_misc_train_f1_macro,
                    _misc_time_per_epoch,_misc_size_subgraph])
    summary_writer = tf.summary.FileWriter(log_dir(FLAGS.train_config,FLAGS.data_prefix,git_branch,git_rev,timestamp), sess.graph)
    # Init variables
    sess.run(tf.global_variables_initializer())
    return model,minibatch, sess, [merged,misc_stats],ph_misc_stat, summary_writer



def train(train_phases,train_params,arch_gcn,model,minibatch,\
            sess,train_stat,ph_misc_stat,summary_writer):
    import time
    avg_time = 0.0
    timing_steps = 0

    saver = tf.train.Saver(var_list=tf.trainable_variables())#global_variables())

    epoch_ph_start = 0
    f1mic_best = 0
    e_best = 0
    time_calc_f1 = 0
    time_qest = 0
    time_train = 0
    time_prepare = 0
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    many_runs_timeline=[]
    for ip,phase in enumerate(train_phases):
        tset_start = time.time()
        minibatch.set_sampler(phase)
        tset_end = time.time()
        time_qest += tset_end-tset_start
        sess.run(model.reset_optimizer_op)
        num_batches = minibatch.num_training_batches()
        printf('START PHASE {:4d}'.format(ip),style='underline')
        for e in range(epoch_ph_start,int(phase['end'])):
            printf('Epoch {:4d}'.format(e),style='bold')
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
                feed_dict, labels = minibatch.feed_dict(phase['dropout'],mode='train')
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
                if not minibatch.batch_num % FLAGS.eval_train_every:
                    t3 = time.time()
                    f1_mic,f1_mac = calc_f1(labels,pred_train,arch_gcn['loss'])
                    #printf("  Iter {:4d}\ttrain loss {:.4f}\tmic {:4f}\tmac {:4f}".format(
                    #    minibatch.batch_num,loss_train,f1_mic,f1_mac))
                    l_loss_tr.append(loss_train)
                    l_f1mic_tr.append(f1_mic)
                    l_f1mac_tr.append(f1_mac)
                    l_size_subg.append(minibatch.size_subgraph)
                    t4 = time.time()
                    time_calc_f1 += t4 - t3
            time_train += time_train_ep
            time_prepare += time_prepare_ep
            loss_val,f1mic_val,f1mac_val,time_eval = \
                    evaluate_full_batch(sess,model,minibatch,many_runs_timeline,mode='val')
            if f1mic_val > f1mic_best:
                f1mic_best = f1mic_val
                e_best = e
                tsave=time.time()
                if not os.path.exists(FLAGS.log_dir+'/models'):
                    os.makedirs(FLAGS.log_dir+'/models')
                print('  Saving models ...')
                savepath = saver.save(sess, '{}/models/saved_model_{}.chkpt'.format(FLAGS.log_dir,timestamp),write_meta_graph=False,write_state=False)
            printf(' TRAIN (Ep avg): loss = {:.4f}\tmic = {:.4f}\tmac = {:.4f}\ttrain time = {:.4f} sec'.format(f_mean(l_loss_tr),f_mean(l_f1mic_tr),f_mean(l_f1mac_tr),time_train_ep))
            printf(' VALIDATION:     loss = {:.4f}\tmic = {:.4f}\tmac = {:.4f}'.format(loss_val,f1mic_val,f1mac_val),style='yellow')

            if FLAGS.tensorboard:
                misc_stat = sess.run([train_stat[1]],feed_dict={\
                                        ph_misc_stat['val_f1_micro']: f1mic_val,
                                        ph_misc_stat['val_f1_macro']: f1mac_val,
                                        ph_misc_stat['train_f1_micro']: f_mean(l_f1mic_tr),
                                        ph_misc_stat['train_f1_macro']: f_mean(l_f1mac_tr),
                                        ph_misc_stat['time_per_epoch']: time_train_ep+time_prepare_ep,
                                        ph_misc_stat['size_subgraph']: f_mean(l_size_subg)})
                # tensorboard visualization
                summary_writer.add_summary(_, e)
                summary_writer.add_summary(misc_stat[0], e)
        epoch_ph_start = int(phase['end'])
    printf("Optimization Finished!",style='yellow')
    timelines = TimeLiner()
    for tl in many_runs_timeline:
        timelines.update_timeline(tl)
    timelines.save('timeline.json')
    saver.restore(sess, '{}/models/saved_model_{}.chkpt'.format(FLAGS.log_dir,timestamp))
    loss_val, f1mic_val, f1mac_val, duration = evaluate_full_batch(sess,model,minibatch,many_runs_timeline,mode='val')
    printf("Full validation (Epoch {:4d}): \n  F1_Micro = {:.4f}\tF1_Macro = {:.4f}".format(e_best,f1mic_val,f1mac_val),style='red')
    loss_test, f1mic_test, f1mac_test, duration = evaluate_full_batch(sess,model,minibatch,many_runs_timeline,mode='test')
    printf("Full test stats: \n  F1_Micro = {:.4f}\tF1_Macro = {:.4f}".format(f1mic_test,f1mac_test),style='red')
    printf('Total training time: {:6.2f} sec'.format(time_train),style='red')
    ret = {'loss_val_opt':loss_val,'f1mic_val_opt':f1mic_val,'f1mac_val_opt':f1mac_val,\
            'loss_test_opt':loss_test,'f1mic_test_opt':f1mic_test,'f1mac_test_opt':f1mac_test,\
            'epoch_best':e_best,
            'time_train': time_train}
    return


########
# MAIN #
########

def train_main(argv=None):
    train_params,train_phases,train_data,arch_gcn = parse_n_prepare(FLAGS)
    model,minibatch,sess,train_stat,ph_misc_stat,summary_writer = prepare(train_data,train_params,arch_gcn)
    ret = train(train_phases,train_params,arch_gcn,model,minibatch,sess,train_stat,ph_misc_stat,summary_writer)
    return ret


if __name__ == '__main__':
    tf.app.run(main=train_main)

