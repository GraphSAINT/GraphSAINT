import numpy as np
import json
import pdb
import scipy.sparse
from sklearn.preprocessing import StandardScaler
import os
import yaml
import scipy.sparse as sp
from graphsaint.globals import _ACT


def load_data(prefix, normalize=True):
    adj_full = scipy.sparse.load_npz('./{}/adj_full.npz'.format(prefix)).astype(np.bool)
    adj_train = scipy.sparse.load_npz('./{}/adj_train.npz'.format(prefix)).astype(np.bool)
    role = json.load(open('./{}/role.json'.format(prefix)))
    feats = np.load('./{}/feats.npy'.format(prefix))
    class_map = json.load(open('./{}/class_map.json'.format(prefix)))
    class_map = {int(k):v for k,v in class_map.items()}
    assert len(class_map) == feats.shape[0]
    # ---- normalize feats ----
    train_nodes = np.array(list(set(adj_train.nonzero()[0])))
    train_feats = feats[train_nodes]
    scaler = StandardScaler()
    scaler.fit(train_feats)
    feats = scaler.transform(feats)
    # -------------------------
    return adj_full, adj_train, feats, class_map, role


def process_graph_data(adj_full, adj_train, feats, class_map, role):
    """
    setup vertex property map for output classes, train/val/test masks, and feats
    INPUT:
        G           graph-tool graph, full graph including training,val,testing
        feats       ndarray of shape |V|xf
        class_map   dictionary {vertex_id: class_id}
        val_nodes   index of validation nodes
        test_nodes  index of testing nodes
    OUTPUT:
        G           graph-tool graph unchanged
        role        array of size |V|, indicating 'train'/'val'/'test'
        class_arr   array of |V|x|C|, converted by class_map
        feats       array of features unchanged
    """
    num_vertices = adj_full.shape[0]
    if isinstance(list(class_map.values())[0],list):
        num_classes = len(list(class_map.values())[0])
        class_arr = np.zeros((num_vertices, num_classes))
        for k,v in class_map.items():
            class_arr[k] = v
    else:
        num_classes = max(class_map.values()) - min(class_map.values()) + 1
        class_arr = np.zeros((num_vertices, num_classes))
        offset = min(class_map.values())
        for k,v in class_map.items():
            class_arr[k][v-offset] = 1
    return adj_full, adj_train, feats, class_arr, role


def parse_layer_yml(dims_str):
    dims_layer = [int(d.split('-')[0]) for d in dims_str]
    order_layer = [int(d.split('-')[1]) for d in dims_str]
    act_layer = [_ACT[d.split('-')[3]] for d in dims_str]
    bias_layer = [d.split('-')[2]=='b' for d in dims_str]
    norm_layer = [d.split('-')[2]=='n' for d in dims_str]
    aggr_layer = [d.split('-')[4] for d in dims_str]
    return dims_layer,order_layer,act_layer,bias_layer,norm_layer,aggr_layer



def get_model_weights(sess,model,dims_gcn):
    gcn_model = model.gcn_model
    num_layer = len(dims_gcn)-1
    dim_out,order,_,bias,norm,__ = parse_layer_yml(dims_gcn[:-1])
    key_weight = list()
    key_bias = list()
    key_norm_off = list()
    key_norm_sca = list()
    if gcn_model == 'gs_mean':
        for l in range(num_layer):
            key_weight.append('neigh_weights_{}'.format(l))
            key_weight.append('self_weights_{}'.format(l))
            if bias[l]:
                key_bias.append('bias_{}'.format(l))
            if norm[l]:
                key_norm_off.append('offset_{}'.format(l))
                key_norm_sca.append('scale_{}'.format(l))
    elif gcn_model == 'gsaint':
        for l in range(num_layer):
            if dim_out[l] == 0:
                continue
            for o in range(order[l]+1):
                key_weight.append('order{}_weights_{}'.format(o,l))
            if bias[l]:
                for o in range(order[l]+1):
                    key_bias.append('order{}_bias_{}'.format(o,l))
            if norm[l]:
                for o in range(order[l]+1):
                    key_norm_off.append('order{}_offset_{}'.format(o,l))
                    key_norm_sca.append('order{}_scale_{}'.format(o,l))
    out_weights = dict()
    for k in key_weight:
        layer = int(k.split('_')[-1])
        out_weights[k] = sess.run(model.aggregators[layer].vars['_'.join(k.split('_')[:-1])])
    for b in key_bias:
        layer = int(b.split('_')[-1])
        out_weights[b] = sess.run(model.aggregators[layer].vars['_'.join(b.split('_')[:-1])])
    for f in key_norm_off:
        layer = int(f.split('_')[-1])
        out_weights[f] = sess.run(model.aggregators[layer].vars['_'.join(f.split('_')[:-1])])
    for s in key_norm_sca:
        layer = int(s.split('_')[-1])
        out_weights[s] = sess.run(model.aggregators[layer].vars['_'.join(s.split('_')[:-1])])
    out_weights['dense_weight'] = sess.run(model.node_pred.vars['weights'])
    out_weights['dense_bias'] = sess.run(model.node_pred.vars['bias'])
    return out_weights


def reload_model_weights(sess,model,weights_best):
    for k,v in weights_best.items():
        print('weight item {}. shape {}'.format(k,v.shape))
        if k.split('_')[0] == 'dense':
            if k.split('_')[-1] == 'weight':
                sess.run(model.node_pred.vars['weights'].assign(v))
            else:
                assert k.split('_')[-1] == 'bias'
                sess.run(model.node_pred.vars['bias'].assign(v))
        else:
            lid = int(k.split('_')[-1])
            _name = '_'.join(k.split('_')[:-1])
            sess.run(model.aggregators[lid].vars[_name].assign(v))
  

def save_model_weights(weights, data_name, epoch, f_train_config):
    if f_train_config == '':
        return
    import pickle
    outf = "./models/{data}-{method}-{epoch}.pkl"
    with open(outf.format(data=data_name,method=f_train_config.split('/')[-1].split('.')[-2],epoch=epoch),'wb') as fout:
        pickle.dump(weights,fout)


def parse_n_prepare(flags,train_config=None):
    if train_config is None:
        with open(flags.train_config) as f_train_config:
            train_config = yaml.load(f_train_config)
    dims_gcn = train_config['network']
    train_phases = train_config['phase']
    train_params = train_config['params'][0]
    for ph in train_phases:
        assert 'end' in ph
        assert 'dropout' in ph
        assert 'sampler' in ph
    print("Loading training data..")
    temp_data = load_data(flags.data_prefix)
    train_data = process_graph_data(*temp_data)
    print("Done loading training data..")
    return train_params,train_phases,train_data,dims_gcn





def log_dir(train_params,dims,f_train_config,prefix,git_branch,git_rev,timestamp):
    import getpass
    log_dir = "/raid/users/"+getpass.getuser()+"/tf_log/" + prefix.split("/")[-1] + 'NeurIPS'
    log_dir += "/{ts}-{model}-{gitrev:s}-{layer}/".format(
            model=train_params['model'],
            gitrev=git_rev.strip(),
            layer='-'.join(dims),
            ts=timestamp)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if f_train_config != '':
        from shutil import copyfile
        copyfile(f_train_config,'{}/{}'.format(log_dir,f_train_config.split('/')[-1]))
    return log_dir

def sess_dir(train_params,dims,train_config,prefix,git_branch,git_rev,timestamp):
    import getpass
    log_dir = "saved_models/" + prefix.split("/")[-1]
    log_dir += "/{ts}-{model}-{gitrev:s}-{layer}/".format(
            model=train_params['model'],
            gitrev=git_rev.strip(),
            layer='-'.join(dims),
            ts=timestamp)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return sess_dir


def adj_norm(adj,norm_adj='sym'):
    """
    Normalize adj according to two methods: symmetric normalization and rw normalization.
    sym norm is used in the original GCN paper (kipf)
    rw norm is used in graphsage and some other variants.

    # Procedure: 
    #       1. adj add self-connection --> adj'
    #       2. D' deg matrix from adj'
    #       3. norm by (D')^(-1/2) x adj' x (D')^(-1/2)
    """
    diag_shape = (adj.shape[0],adj.shape[1])
    if norm_adj == 'sym':
        adj_I = (adj + sp.eye(adj.shape[0])).astype(np.bool)
        D_I = np.array(adj_I.sum(1).flatten())
        D_I = sp.dia_matrix((D_I**-.5,0),shape=diag_shape)
        adj_norm = D_I@adj_I@D_I
    elif norm_adj == 'rw':
        D = adj.sum(1).flatten()
        norm_diag = sp.dia_matrix((1/D,0),shape=diag_shape)
        adj_norm = norm_diag.dot(adj)
    return adj_norm


def deg_mat_inv(adj):
    """
    return inverse degree matrix of the subg adj.
    -- simply sum the rows
    """
    diag_shape = (adj.shape[0],adj.shape[1])
    D = adj.sum(1).flatten()
    return sp.dia_matrix((1/D,0),shape=diag_shape)
