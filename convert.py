import scipy.sparse as sp
import numpy as np
import networkx as nx
import sys
import json
import os
from networkx.readwrite import json_graph

dataset_str=sys.argv[1]
baseline_str='data.ignore/'+dataset_str+'/'
dataset_str='data/'+dataset_str+'/'
if not os.path.exists(baseline_str[:-1]):
    os.mkdir(baseline_str[:-1])

# G.json
adj_full=sp.load_npz(dataset_str+'adj_full.npz')
G=nx.from_scipy_sparse_matrix(adj_full)
print('nx: finish load graph')
data=json_graph.node_link_data(G)
role=json.load(open(dataset_str+'role.json','r'))
te=set(role['te'])
va=set(role['va'])
for node in data['nodes']:
    node['test']=False
    node['val']=False
    if node['id'] in te:
        node['test']=True
    elif node['id'] in va:
        node['val']=True
for edge in data['links']:
    del edge['weight']
    edge['target']=int(edge['target'])
with open(baseline_str+'G.json','w') as f:
    json.dump(data,f)
 

# id_map.json
id_map={}
for i in range(G.number_of_nodes()):
    id_map[str(i)]=i
with open(baseline_str+'id_map.json','w') as f:
    json.dump(id_map,f)

# feats.npy
feats=np.load(dataset_str+'feats.npy')
np.save(baseline_str+'feats.npy',feats)

# class_map.json
class_map=json.load(open(dataset_str+'class_map.json','r'))
for k,v in class_map.items():
    class_map[k]=v
with open(baseline_str+'class_map.json','w') as f:
    json.dump(class_map,f)
