import scipy.sparse as sp
import numpy as np
import networkx as nx
import sys
import json
import os
from networkx.readwrite import json_graph

if len(sys.argv)==1:
    datasets=['ppi','flickr','reddit','amazon']
else:
    datasets=[sys.argv[1]]

for dataset in datasets:
    print('start ',dataset)
    baseline_str='../data.cpp/'+dataset+'/'
    dataset_str='../data/'+dataset+'/'
    if not os.path.exists(baseline_str[:-1]):
        os.mkdir(baseline_str[:-1])

    adj_full=sp.load_npz(dataset_str+'adj_full.npz')
    adj_train=sp.load_npz(dataset_str+'adj_train.npz')
    role=json.load(open(dataset_str+'role.json','r'))
    feats=np.load(dataset_str+'feats.npy')
    class_map=json.load(open(dataset_str+'class_map.json','r'))
    if dataset=='reddit':
        class_map_np=np.zeros((len(class_map),41))
        for i in range(len(class_map)):
            class_map_np[i,class_map[str(i)]]=1
    elif dataset=='flickr':
        class_map_np=np.zeros((len(class_map),7))
        for i in range(len(class_map)):
            class_map_np[i,class_map[str(i)]]=1
    else:
        class_map_np=np.zeros((len(class_map),len(class_map['0'])))
        for i in range(len(class_map)):
            class_map_np[i]=class_map[str(i)]
    # import pdb; pdb.set_trace();

    dims=np.zeros((11,))
    dims[0]=adj_train.indices.shape[0]
    dims[1]=adj_train.indptr.shape[0]
    dims[2]=adj_full.indices.shape[0]
    dims[3]=adj_full.indptr.shape[0]
    dims[4]=len(role['tr'])
    dims[5]=len(role['te'])
    dims[6]=len(role['va'])
    dims[7]=feats.shape[0]
    dims[8]=feats.shape[1]
    dims[9]=class_map_np.shape[0]
    dims[10]=class_map_np.shape[1]
    dims.astype(np.int32).tofile(baseline_str+'dims.bin')

    adj_train.indices.astype(np.int32).tofile(baseline_str+'adj_train_indices.bin')
    adj_train.indptr.astype(np.int32).tofile(baseline_str+'adj_train_indptr.bin')
    adj_full.indices.astype(np.int32).tofile(baseline_str+'adj_full_indices.bin')
    adj_full.indptr.astype(np.int32).tofile(baseline_str+'adj_full_indptr.bin')
    np.array(role['tr']).astype(np.int32).tofile(baseline_str+'node_train.bin')
    np.array(role['te']).astype(np.int32).tofile(baseline_str+'node_test.bin')
    np.array(role['va']).astype(np.int32).tofile(baseline_str+'node_val.bin')

    from sklearn.preprocessing import StandardScaler
    train_feats=feats[np.array(role['tr'])]
    scaler=StandardScaler()
    scaler.fit(train_feats)
    feats=scaler.transform(feats)
    feats.T.astype(np.float64).tofile(baseline_str+'feats_norm_col.bin')

    class_map_np.T.astype(np.float64).tofile(baseline_str+'labels_col.bin')