import json
import sys
import os
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from ogb.nodeproppred import PygNodePropPredDataset

"""
Run this script to convert the graph from the open graph benchmark format
to the GraphSAINT format.

Right now, ogbn-products and ogbn-arxiv can be converted by this script.
"""


dataset = PygNodePropPredDataset(name=sys.argv[1])
split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
graph = dataset[0]
num_node = graph.y.shape[0]
# import pdb; pdb.set_trace()

save_dir = './data/'+sys.argv[1]+'/'
try:
    os.mkdir(save_dir)
except OSError as error:
    print(error)

# feats.npy
feats = graph.x.numpy()
np.save(save_dir+'feats.npy',feats)

# role.json
role = dict()
role['tr'] = train_idx.numpy().tolist()
role['va'] = valid_idx.numpy().tolist()
role['te'] = test_idx.numpy().tolist()
with open(save_dir+'role.json','w') as f:
    json.dump(role, f)

# class_map.json
class_map = dict()
for i in range(num_node):
    class_map[str(i)] = int(graph.y[i])
with open(save_dir + 'class_map.json', 'w') as f:
    json.dump(class_map, f)

# adj_*.npz
train_idx_set = set(train_idx.numpy().tolist())
test_idx_set = set(test_idx.numpy().tolist())
edge_index = graph.edge_index.numpy()
row_full = edge_index[0]
col_full = edge_index[1]
row_train = []
col_train = []
row_val = []
col_val = []
for i in tqdm(range(row_full.shape[0])):
    if row_full[i] in train_idx_set and col_full[i] in train_idx_set:
        row_train.append(row_full[i])
        col_train.append(col_full[i])
        row_val.append(row_full[i])
        col_val.append(col_full[i])
    elif not (row_full[i] in test_idx_set or col_full[i] in test_idx_set):
        row_val.append(row_full[i])
        col_val.append(col_full[i])
row_train = np.array(row_train)
col_train = np.array(col_train)
row_val = np.array(row_val)
col_val = np.array(col_val)
dtype = np.bool

adj_full = sp.coo_matrix(
    (
        np.ones(row_full.shape[0], dtype=dtype),
        (row_full, col_full),
    ),
    shape=(num_node, num_node)
).tocsr()

adj_train = sp.coo_matrix(
    (
        np.ones(row_train.shape[0], dtype=dtype),
        (row_train, col_train),
    ),
    shape=(num_node, num_node)
).tocsr()

adj_val = sp.coo_matrix(
    (
        np.ones(row_val.shape[0], dtype=dtype),
        (row_val, col_val),
    ),
    shape=(num_node, num_node)
).tocsr()

# import pdb; pdb.set_trace()
print('adj_full  num edges:', adj_full.nnz)
print('adj_val   num edges:', adj_val.nnz)
print('adj_train num edges:', adj_train.nnz)
sp.save_npz(save_dir+'adj_full.npz', adj_full)
sp.save_npz(save_dir+'adj_train.npz', adj_train)
# adj_val not used in GraphSAINT
sp.save_npz(save_dir+'adj_val.npz', adj_val)
