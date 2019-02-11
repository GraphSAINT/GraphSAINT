import tensorflow as tf
import scipy.sparse
import numpy as np
import sys
import pdb

np.random.seed(123)

data = sys.argv[1]
len_feat = int(sys.argv[2])

adj = scipy.sparse.load_npz('../data/{}/adj_full.npz'.format(data))
V = adj.shape[0]
stride = int(sys.argv[3])


with tf.device('/gpu:0'):
    ph_adj_gpu = tf.sparse_placeholder(tf.float32,name='adj_gpu')
    ph_feat_gpu = tf.placeholder(tf.float32,shape=(V,len_feat))
    ph_stride = tf.placeholder(tf.int32)
    f_cond = lambda i,r: i<len_feat
    f_body = lambda i,r: (i+ph_stride,tf.concat(\
        [r,tf.sparse_tensor_dense_matmul(ph_adj_gpu,ph_feat_gpu[:,i:i+ph_stride])],axis=1))
    result_gpu = tf.while_loop(f_cond,f_body,\
        (ph_stride,tf.sparse_tensor_dense_matmul(ph_adj_gpu,ph_feat_gpu[:,:ph_stride])))
    ######################
    #ret_l = []
    #for s in range(0,len_feat,stride):
    #    ret_l.append(tf.sparse_tensor_dense_matmul(ph_adj_gpu,ph_feat_gpu[:,s:s+stride]))
    #result_gpu = tf.concat(ret_l,axis=1)

with tf.device('/cpu:0'):
    ph_adj_cpu = tf.sparse_placeholder(tf.float32,name='adj_cpu')
    ph_feat_cpu = tf.placeholder(tf.float32,shape=(V,len_feat))
    result_cpu = tf.sparse_tensor_dense_matmul(ph_adj_cpu,ph_feat_cpu)

feed_adj = tf.SparseTensorValue(np.column_stack(adj.nonzero()),adj.data,adj.shape)
feed_feat = np.random.rand(V,len_feat)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
ret_gpu = sess.run(result_gpu,feed_dict={ph_adj_gpu:feed_adj,ph_feat_gpu:feed_feat,ph_stride:10})[1].astype(np.float64)
ret_cpu = sess.run(result_cpu,feed_dict={ph_adj_cpu:feed_adj,ph_feat_cpu:feed_feat}).astype(np.float64)

print("finished\n\tfeat sum is: {}\n\tresult sum is: {}\n\tresult gnd is: {}".format(feed_feat.sum(),ret_gpu.sum(),ret_cpu.sum()))
