from graphsaint.adj_misc import *
import tensorflow as tf
import scipy.sparse
import sys,pdb,os
import time
import numpy as np


if __name__ == "__main__":
    num_proc = int(sys.argv[1])
    adj = scipy.sparse.load_npz(sys.argv[2])
    scale = 1/np.array(adj.sum(1),dtype=np.float32).flatten()
    new_adj_data = np.ones(adj.data.size,dtype=np.float32)
    t1 = time.time()
    new_adj_data = adj_norm_cython(adj.indptr,adj.indices,new_adj_data,scale,num_proc=num_proc)
    t2 = time.time()
    adj_norm_test = scipy.sparse.csr_matrix((new_adj_data,adj.indices,adj.indptr),shape=adj.shape)
    t3 = time.time()
    print('time for {} processors: {:6.4f}, {:6.4f}'.format(num_proc,t2-t1,t3-t2))
    norm_diag = scipy.sparse.dia_matrix((scale,0),shape=adj.shape)
    adj_norm_golden = norm_diag@adj
    assert len(np.where(adj_norm_golden.data-adj_norm_test.data>1e-8)[0])==0
    print('test passed')
