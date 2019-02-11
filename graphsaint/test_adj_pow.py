from graphsaint.adj_pow import *
import tensorflow as tf
import scipy.sparse
import sys,pdb,os
import time
import numpy as np

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"
GPU_MEM_FRACTION = 0.8


POW = 2

np.random.seed(0)

def gen_rand_testcase(dim,density):
    scaling = 10
    ret_sp = scipy.sparse.random(dim,dim,density=density,format='csr')
    ret_sp.data = (10*ret_sp.data).astype(np.int32)
    ret_ds = (ret_sp.toarray()).astype(np.int32)
    ret_ds = ret_ds + ret_ds.T
    ret_sp = scipy.sparse.csr_matrix(ret_ds)
    return ret_sp,ret_ds



if __name__ == "__main__":
    num_proc = int(sys.argv[1])
    method = sys.argv[2]
    if sys.argv[3].split('.')[-1] == "npz":
        verify_result = False
        adj = scipy.sparse.load_npz(sys.argv[3])
        is_adj_val1 = True
    else:
        verify_result = True
        dim = int(sys.argv[3])
        density = float(sys.argv[4])
        adj,adj_dense = gen_rand_testcase(dim,density)
        is_adj_val1 = False
    t1 = time.time()
    ret = get_adj_pow(adj,POW,num_proc=num_proc,method=method,is_adj_val1=is_adj_val1)
    t2 = time.time()
    print("time calling cython function: ",t2-t1)
    ret = scipy.sparse.csr_matrix((ret[2],ret[1],ret[0]),shape=(len(ret[0])-1,len(ret[0])-1))
    t3 = time.time()
    print("time converting to scipy sparse format: ",t3-t2)
    print("return adj degree: ",ret.size/ret.shape[0])
    if verify_result:
        ret_dense = ret.toarray()
        tf_ret_dense = tf.constant(adj_dense,shape=adj_dense.shape)
        ret_golden = tf.eye(adj_dense.shape[0],dtype=tf.int32)
        for p in range(POW):
            ret_golden = tf.linalg.matmul(ret_golden,tf_ret_dense)
        sess = tf.Session()
        ret_golden = sess.run(ret_golden)
        assert (ret_golden-ret_dense).sum() == 0
        print("test passed.")
    else:
        # save adj
        new_adj = '{}_hop{}.npz'.format(sys.argv[3].split('.npz')[0],POW)
        scipy.sparse.save_npz(new_adj,ret)
