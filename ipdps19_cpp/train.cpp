#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "global.h"
#include "util.h"
#include "init.h"
#include "sample.h"
#include "operation.h"
#ifdef USE_MKL
    #include "mkl.h"
#endif
#include "optm.h"
#include "layer.h"

using namespace std;

map<char,double> time_ops;
int dim_init, num_cls;      // dimension of initial node features; num of output classes
int num_layer, size_subg, size_frontier, method_sample(DEFAULT_METHOD_SAMPLE);
double lr;  // learning rate
bool is_sigmoid;
int dim_hid;
char *data;
int num_itr;
int num_thread;

void forward_GNN(Model &model, s_data2d_sp &adj, 
                 s_idx1d_ds &v_subg, s_data2d_ds &feat_fullg, 
                 s_idx2d_ds &label_subg, s_stat_acc &loss_acc, s_idx1d_ds v_masked=s_idx1d_ds()) {
    int num_layer = model.layer_SAGE.size();
    // setup layer-1 input
    lookup_feats(v_subg, feat_fullg, model.layer_SAGE[0].feat_in);
    for (int l=0; l<num_layer-1; l++) {
        model.layer_SAGE[l].forward(adj, model.layer_SAGE[l+1].feat_in);
    }
    model.layer_SAGE[num_layer-1].forward(adj, model.layer_l2norm.feat_in);
    model.layer_l2norm.forward(model.layer_dense.feat_in, adj.num_v);
    model.layer_dense.forward(model.layer_loss.feat_in, adj.num_v);
    model.layer_loss.forward(label_subg, loss_acc, v_masked);
}

void backward_GNN(Model &model, s_data2d_sp &adj, s_data2d_sp &adj_trans, 
                  s_idx2d_ds &label_subg, s_stat_acc &loss_acc) {
    int num_layer = model.layer_SAGE.size();
    model.layer_loss.backward(label_subg, model.layer_dense.grad_in);
    model.layer_dense.backward(model.layer_l2norm.grad_in);
    model.layer_l2norm.backward(model.layer_SAGE[num_layer-1].grad_in);
    for (int l=num_layer-1; l>0; l--) {
        model.layer_SAGE[l].backward(adj, adj_trans, model.layer_SAGE[l-1].grad_in);
    }
    s_data2d_ds dummy = s_data2d_ds();
    model.layer_SAGE[0].backward(adj, adj_trans, dummy);
}


int main(int argc, char* argv[]) {
    parse_args(argc,argv,data,num_itr,num_thread,num_layer,
               size_subg,size_frontier,dim_hid,lr,is_sigmoid);

    s_data2d_sp adj_full,adj_train;
    s_idx1d_ds node_all,node_train,node_val,node_test;
    s_data2d_ds input;          // input is input node feat for the full graph
    s_idx2d_ds label_true;
    // load_data will initialize all the data struct inside
    load_data(data,adj_full,adj_train,node_all,node_train,node_val,node_test,input,label_true);
    dim_init = input.dim2;
    num_cls = label_true.dim2;
    /* *************
     * INIT SAMPLERS
     * ************* */
    s_data2d_sp *subgs = new s_data2d_sp[num_thread];
    s_data2d_sp *subgs_trans = new s_data2d_sp[num_thread];
    s_idx1d_ds *subgs_v = new s_idx1d_ds[num_thread];
    s_data2d_sp subg_cur, subg_trans_cur, subg_eval;
    s_idx1d_ds subg_v_cur, subg_v_eval;
    auto sample = sample_frontier;
    switch(method_sample) {
        case SAMPLE_FRONTIER:
            sample = sample_frontier;
            break;
        default:
            sample = sample_frontier;
    }

    omp_set_nested(true);
    // setup labels
    s_idx2d_ds label_subg(size_subg, label_true.dim2);
    s_idx2d_ds label_true_val(node_val.dim1, label_true.dim2);
    s_idx2d_ds label_true_test(node_test.dim1, label_true.dim2);
    lookup_labels(node_val, label_true, label_true_val);
    lookup_labels(node_test, label_true, label_true_test);
    int num_subg_remain = 0;
    // ======================
    // SETUP THREADS 
    // ======================
    omp_set_num_threads(num_thread);
    omp_set_dynamic(0);
#ifdef USE_MKL
    mkl_set_num_threads(num_thread);
    mkl_set_dynamic(0);
#endif

    #pragma omp parallel default (shared) 
    {   
        int ompTid = omp_get_thread_num();
        int proc_id = ompTid;
        bind_to_proc(proc_id);
    }
    /* ***********
     * BUILD MODEL
     * *********** */
    Model model;
    int size_g = adj_full.num_v;        // this is to support full batch inference on val / test nodes
    for (int l=0; l<num_layer; l++) {
        int dim_weight_in = (l==0) ? dim_init : dim_hid;
        bool is_act = l==num_layer-1?false:true;
        model.layer_SAGE.push_back(Layer_SAGE(l,size_g,num_thread,is_act,dim_hid,dim_weight_in,lr));
    }
    model.layer_l2norm = Layer_l2norm(0,size_g,num_thread,dim_hid);
    model.layer_dense = Layer_dense(0,size_g,num_thread,num_cls,dim_hid,lr);
    model.layer_loss = Layer_loss(0,size_g,num_thread,num_cls,is_sigmoid);
    // ========================
    // TRAIN LOOP 
    // ========================
    time_ops[OP_DENSE] = 0.;
    time_ops[OP_SPARSE] = 0.;
    time_ops[OP_RELU] = 0.;
    time_ops[OP_NORM] = 0.;
    time_ops[OP_LOOKUP] = 0.;
    time_ops[OP_BIAS] = 0.;
    time_ops[OP_MASK] = 0.;
    time_ops[OP_REDUCE] = 0.;
    time_ops[OP_SIGMOID] = 0.;
    time_ops[OP_SOFTMAX] = 0.;

    s_stat_acc loss_acc;
    double time_ep = 0;
    for (int itr=0; itr<num_itr; itr++) {
        // validation
        if (itr%EVAL_INTERVAL==0 && itr!=0) {
            forward_GNN(model, adj_full, node_all, input, 
                        label_true_val, loss_acc, node_val);
            printf("Evaluation f1_mic: %f, f1_mac: %f\n", loss_acc.f1_mic, loss_acc.f1_mac);
        }
        // training
        printf("============\nITERATION %d\n============\n",itr);
        if (num_subg_remain == 0) {
            sample(adj_train,node_train,subgs,subgs_v,size_subg,DEFAULT_SIZE_FRONTIER);
            #pragma omp parallel for
            for (int i=0; i<num_thread; i++) {
                subgs_trans[i].num_v = subgs[i].num_v;
                subgs_trans[i].num_e = subgs[i].num_e;
                subgs_trans[i].indptr = subgs[i].indptr;
                subgs_trans[i].indices = subgs[i].indices;
                if (subgs_trans[i].arr != NULL) {_free(subgs_trans[i].arr);}
                subgs_trans[i].arr = (t_data*)_malloc(subgs[i].num_e*sizeof(t_data));
                transpose_adj(subgs[i],subgs_trans[i].arr);
            }
            num_subg_remain = num_thread;
        }
        num_subg_remain--;
        subg_cur = subgs[num_subg_remain];
        subg_trans_cur = subgs_trans[num_subg_remain];
        subg_v_cur = subgs_v[num_subg_remain];
        lookup_labels(subg_v_cur, label_true, label_subg);
        forward_GNN(model, subg_cur, subg_v_cur, input, label_subg, loss_acc);
        backward_GNN(model, subg_cur, subg_trans_cur, label_subg, loss_acc);
        printf("Training itr %d f1_mic: %f, f1_mac: %f\n", itr, loss_acc.f1_mic, loss_acc.f1_mac);
    }
    printf("--------------------\n");
    printf("DENSE time: %lf\n", time_ops[OP_DENSE]);
    printf("SPARSE time: %lf\n", time_ops[OP_SPARSE]);
    printf("RELU time: %lf\n", time_ops[OP_RELU]);
    printf("NORM time: %lf\n", time_ops[OP_NORM]);
    printf("LOOKUP time: %lf\n", time_ops[OP_LOOKUP]);
    printf("BIAS time: %lf\n", time_ops[OP_BIAS]);
    printf("MASK time: %lf\n", time_ops[OP_MASK]);
    printf("REDUCE time: %lf\n", time_ops[OP_REDUCE]);
    printf("SIGMOID time: %lf\n", time_ops[OP_SIGMOID]);
    printf("SOFTMAX time: %lf\n", time_ops[OP_SOFTMAX]);
    printf("--------------------\n");
    forward_GNN(model, adj_full, node_all, input,
                label_true_test, loss_acc, node_test);
    printf("Testing f1_mic: %f, f1_mac: %f\n", loss_acc.f1_mic, loss_acc.f1_mac);
}


