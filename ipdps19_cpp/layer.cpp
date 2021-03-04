#include "global.h"
#include "operation.h"
#include "util.h"
#include <omp.h>
#include <cassert>
#include <cstring>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#ifdef USE_MKL
#include "mkl.h"
#endif
#include "layer.h"
#include "init.h"
#include <algorithm>
#include <utility>
#include <vector>
#include <limits>


Layer_SAGE::Layer_SAGE() {
    id_layer = size_subg = num_thread = -1;
    dim_weight_out = dim_weight_in = -1;
}

Layer_SAGE::Layer_SAGE(int id_layer_, int size_subg_, int num_thread_, bool is_act_, int dim_weight_out_, int dim_weight_in_, double lr_)
{
    id_layer = id_layer_;
    size_subg = size_subg_;
    num_thread = num_thread_;
    dim_weight_out = dim_weight_out_;
    dim_weight_in = dim_weight_in_==-1 ? dim_weight_out_:dim_weight_in_;
    weight_neigh = s_data2d_ds(dim_weight_in, dim_weight_out/2);
    d_weight_neigh = s_data2d_ds(dim_weight_in, dim_weight_out/2);
    weight_self = s_data2d_ds(dim_weight_in, dim_weight_out/2);
    d_weight_self = s_data2d_ds(dim_weight_in, dim_weight_out/2);
    bias = s_data1d_ds(dim_weight_out);
    d_bias = s_data1d_ds(dim_weight_out);
    // init weight
    init_glorot(weight_self);
    init_glorot(weight_neigh);
    init_zero(bias);
    // determine the order of matrix chain multiplication (for both forward and backward)
    is_order_AX_W = dim_weight_in < dim_weight_out / 2 ? true : false;
    feat_aggr = s_data2d_ds(size_subg, is_order_AX_W ? dim_weight_in : dim_weight_out/2);
    feat_in = s_data2d_ds(size_subg, dim_weight_in);
    grad_in = s_data2d_ds(size_subg, dim_weight_out);
    is_act = is_act_;
    if (is_act) {
        mask = s_idx1d_ds(size_subg*dim_weight_out);
    } else {
        mask = s_idx1d_ds();
    }
    // init optm
    std::vector<std::pair<int,int>> dims2d{std::make_pair(weight_self.dim1,weight_self.dim2),
                                 std::make_pair(weight_neigh.dim1,weight_neigh.dim2)};
    std::vector<int> dims1d{bias.dim1};
    optm = ADAM(dims2d,dims1d,lr_=lr_);
    weights.push_back(&weight_self);
    weights.push_back(&weight_neigh);
    d_weights.push_back(&d_weight_self);
    d_weights.push_back(&d_weight_neigh);
    biases.push_back(&bias);
    d_biases.push_back(&d_bias);
}

void Layer_SAGE::update_size_subg(int size_subg_, s_data2d_ds &feat_out) {
    size_subg = size_subg_;
    feat_aggr.dim1 = size_subg;
    feat_in.dim1 = size_subg;
    grad_in.dim1 = size_subg;
    feat_out.dim1 = size_subg;
}

/*
 * Assume before calling forward, feat_in has already been filled in. 
 * feat_out should be the feat_in of next layer
 */
void Layer_SAGE::forward(s_data2d_sp subg, s_data2d_ds &feat_out)
{
    update_size_subg(subg.num_v, feat_out);
    feat_out.dim1 = subg.num_v;
    // prepare for concatenation
    s_data2d_ds feat_out_part1 = s_data2d_ds();
    s_data2d_ds feat_out_part2 = s_data2d_ds();
    feat_out_part1.dim1 = feat_out.dim1;        // part 1 of concat
    feat_out_part1.dim2 = feat_out.dim2/2;
    feat_out_part1.arr = feat_out.arr;
    feat_out_part2.dim1 = feat_out.dim1;        // part 2 of concat
    feat_out_part2.dim2 = feat_out.dim2/2;
    feat_out_part2.arr = feat_out.arr + feat_out_part1.dim1*feat_out_part1.dim2;
    denseMM(feat_in, weight_self, feat_out_part1);
    // feat_aggr*weight_neigh || feat_in*weight_self + bias
    if (is_order_AX_W) {    // feat aggregation (order 1: (AX)W)
        feat_aggr.dim2 = dim_weight_in;
        sparseMM(subg, feat_in, feat_aggr, num_thread); // update feat_aggr
        denseMM(feat_aggr, weight_neigh, feat_out_part2);
    } else {                // feat aggregation (order 2: A(XW))
        feat_aggr.dim2 = dim_weight_out/2;
        denseMM(feat_in, weight_neigh, feat_aggr);
        sparseMM(subg, feat_aggr, feat_out_part2, num_thread);
    }
    // apply bias
    biasMV(feat_out, bias);
    // activation
    if (is_act) {relu(feat_out, mask, num_thread);}
}

/*
 * Assume before calling backward, grad_in has already been filled in. 
 * grad_out should be the grad_in of previous layer
 * grad_out = d L / d X^(l-1)
 */
void Layer_SAGE::backward(s_data2d_sp subg, s_data2d_sp subg_trans, s_data2d_ds &grad_out)
{
    if (is_act) {maskM(grad_in, mask);}
    // partition grad_in for the part of self and neighbor weights
    s_data2d_ds grad_in_part1 = s_data2d_ds();
    s_data2d_ds grad_in_part2 = s_data2d_ds();
    grad_in_part1.dim1 = grad_in.dim1;
    grad_in_part1.dim2 = grad_in.dim2/2;
    grad_in_part1.arr = grad_in.arr;
    grad_in_part2.dim1 = grad_in.dim1;
    grad_in_part2.dim2 = grad_in.dim2/2;
    grad_in_part2.arr = grad_in.arr + grad_in_part1.dim1*grad_in_part1.dim2;
    denseMM(feat_in, grad_in_part1, d_weight_self, true);
    // In the below calculation, we use feat_aggr to store the temporary result for matrix multiplication. 
    // Therefore, we do not need to allocation any new storage. 
    if (is_order_AX_W) {
        assert(feat_aggr.dim2 == dim_weight_in);        // feat_aggr stores the aggregated, raw feature
        sparseMM(subg, feat_in, feat_aggr, num_thread);                     // feat_aggr = adj X_{in}
        denseMM(feat_aggr, grad_in_part2, d_weight_neigh, true);            // grad_weight = feat_aggr^T grad_in
        if (id_layer > 0) {
            denseMM(grad_in_part2, weight_neigh, feat_aggr, false, true);   // feat_aggr = grad_in W_{neigh}^T
            sparseMM(subg_trans, feat_aggr, grad_out, num_thread);
        }
    } else {
        assert(feat_aggr.dim2 == dim_weight_out/2);     // feat_aggr stores the aggregated, transformed feature
        sparseMM(subg_trans, grad_in_part2, feat_aggr, num_thread);
        denseMM(feat_in, feat_aggr, d_weight_neigh, true);
        if (id_layer > 0) {
            denseMM(feat_aggr, weight_neigh, grad_out, false, true);        // grad_out = (adj^T grad_in) W_{neigh}^T
        }
    }
    if (id_layer > 0) {
        denseMM(grad_in_part1, weight_self, grad_out, false, true, true);   // perform += of grad_out
    }
    // gradient of bias
    reduce_sum(grad_in, d_bias, 0);
    // update weights by gradients
    weights[0] = &weight_self;
    weights[1] = &weight_neigh;
    d_weights[0] = &d_weight_self;
    d_weights[1] = &d_weight_neigh;
    biases[0] = &bias;
    d_biases[0] = &d_bias;
    optm.update(weights,d_weights,biases,d_biases);
}





/* ***********
 * DENSE LAYER
 * *********** */
Layer_dense::Layer_dense() {
    id_layer = size_subg = num_thread = -1;
    dim_weight_out = dim_weight_in = -1;
}
Layer_dense::Layer_dense(int id_layer_, int size_subg_, int num_thread_, int dim_weight_out_, int dim_weight_in_, double lr_)
{
    id_layer = id_layer_;
    size_subg = size_subg_;
    num_thread = num_thread_;
    dim_weight_out = dim_weight_out_;
    dim_weight_in = dim_weight_in_==-1 ? dim_weight_out_:dim_weight_in_;
    weight = s_data2d_ds(dim_weight_in,dim_weight_out);
    d_weight = s_data2d_ds(dim_weight_in,dim_weight_out);
    bias = s_data1d_ds(dim_weight_out);
    d_bias = s_data1d_ds(dim_weight_out);
    // init weights
    init_glorot(weight);
    init_zero(bias);
    feat_in = s_data2d_ds(size_subg,dim_weight_in);
    grad_in = s_data2d_ds(size_subg,dim_weight_out);
    mask = s_idx1d_ds(size_subg*dim_weight_out);
    // init optm
    std::vector<std::pair<int,int>> dims2d{std::make_pair(weight.dim1,weight.dim2)};
    std::vector<int> dims1d{bias.dim1};
    optm = ADAM(dims2d,dims1d,lr_=lr_);
    weights.push_back(&weight);
    d_weights.push_back(&d_weight);
    biases.push_back(&bias);
    d_biases.push_back(&d_bias);
}

void Layer_dense::update_size_subg(int size_subg_, s_data2d_ds &feat_out) {
    size_subg = size_subg_;
    feat_in.dim1 = size_subg;
    grad_in.dim1 = size_subg;
    feat_out.dim1 = size_subg;
}

void Layer_dense::forward(s_data2d_ds &feat_out, int size_subg_) {
    update_size_subg(size_subg_, feat_out);
    feat_out.dim1 = size_subg_;
    denseMM(feat_in, weight, feat_out);
    biasMV(feat_out, bias);
    //relu(feat_out, mask, num_thread);
}

void Layer_dense::backward(s_data2d_ds &grad_out) {
    //maskM(grad_in, mask);
    denseMM(feat_in, grad_in, d_weight, true);
    denseMM(grad_in, weight, grad_out, false, true);
    reduce_sum(grad_in, d_bias, 0);
    weights[0] = &weight;
    d_weights[0] = &d_weight;
    biases[0] = &bias;
    d_biases[0] = &d_bias;
    optm.update(weights,d_weights,biases,d_biases);
}



/* *************
 * L2-NORM LAYER
 * ************* */
Layer_l2norm::Layer_l2norm() {
    id_layer = size_subg = num_thread = -1;
    dim = -1;
}
Layer_l2norm::Layer_l2norm(int id_layer_, int size_subg_, int num_thread_, int dim_)
{
    id_layer = id_layer_;
    size_subg = size_subg_;
    num_thread = num_thread_;
    dim = dim_;
    feat_in = s_data2d_ds(size_subg, dim);
    grad_in = s_data2d_ds(size_subg, dim);
}

void Layer_l2norm::l2norm(s_data2d_ds &feat_out)
{
    double t1 = omp_get_wtime();
    assert(feat_in.dim2 == feat_out.dim2);
    feat_out.dim1 = feat_in.dim1;
    int dim1 = feat_in.dim1;
    int dim2 = feat_in.dim2;
    #pragma omp parallel for
    for (int i=0; i<dim1; i++) {
        float sum = 0;
        for (int j=0; j<dim2; j++) {
            sum += feat_in.arr[i+j*dim1]*feat_in.arr[i+j*dim1];
        }
        sum = sum<1.0e-12?1.0e-12:sum;
        sum = sqrt(sum);
        for (int j=0; j<dim2; j++) {
            feat_out.arr[i+j*dim1] = feat_in.arr[i+j*dim1]/sum;
        }
    }
    double t2 = omp_get_wtime();
    time_ops[OP_NORM] += t2-t1;
}

void Layer_l2norm::update_size_subg(int size_subg_, s_data2d_ds &feat_out) {
    size_subg = size_subg_;
    feat_in.dim1 = size_subg;
    grad_in.dim1 = size_subg;
    feat_out.dim1 = size_subg;
}

void Layer_l2norm::d_l2norm(s_data2d_ds &grad_out)
{
    double t1 = omp_get_wtime();
    assert(feat_in.dim2 == grad_in.dim2 && feat_in.dim1 == grad_in.dim1);
    int dim1 = grad_in.dim1;
    int dim2 = grad_in.dim2;
    #pragma omp parallel for
    for (int i=0; i<dim1; i++) {
        t_data coef0_axis0 = 0, coef1_axis0 = 0;
        t_data sum_x2 = 0;
        for (int j=0; j<dim2; j++) {
            sum_x2 += powf(feat_in.arr[i+j*dim1],2);
            coef0_axis0 -= feat_in.arr[i+j*dim1] * grad_in.arr[i+j*dim1];
        }
        coef1_axis0 = powf(sum_x2, -1.5);
        for (int j=0; j<dim2; j++) {
            grad_out.arr[i+j*dim1] = feat_in.arr[i+j*dim1]*coef0_axis0*coef1_axis0
                        + grad_in.arr[i+j*dim1]*sum_x2*coef1_axis0;
        }
    }
    double t2 = omp_get_wtime();
    time_ops[OP_NORM] += t2-t1;
}

void Layer_l2norm::forward(s_data2d_ds &feat_out, int size_subg_) {
    update_size_subg(size_subg_, feat_out);
    l2norm(feat_out);
}

void Layer_l2norm::backward(s_data2d_ds &grad_out) {
    d_l2norm(grad_out);
}



Layer_loss::Layer_loss() {
    id_layer = size_subg = num_thread = -1;
    num_cls = -1;
}
Layer_loss::Layer_loss(int id_layer_, int size_subg_, int num_thread_, int num_cls_, bool is_sigmoid_)
{
    id_layer = id_layer_;
    size_subg = size_subg_;
    num_thread = num_thread_;
    num_cls = num_cls_;
    is_sigmoid = is_sigmoid_;
    feat_in = s_data2d_ds(size_subg, num_cls);
    pred = s_data2d_ds(size_subg, num_cls);
    pred_bin = s_idx2d_ds(size_subg, num_cls);
}

void Layer_loss::sigmoid(s_idx1d_ds &v_masked) {
    if (v_masked.dim1 > 0) {
        int num_masked = v_masked.dim1;
        #pragma omp parallel for
        for (int i=0; i<num_masked; i++) {
            int vid_masked = v_masked.arr[i];
            for (int j=0; j<num_cls; j++) {
                pred.arr[i+j*num_masked] = 1./(1.+expf(-feat_in.arr[vid_masked+j*feat_in.dim1]));
            }
        }
    } else {
        #pragma omp parallel for
        for (int i=0; i<feat_in.dim1*feat_in.dim2; i++) {
            pred.arr[i] = 1./(1.+expf(-feat_in.arr[i]));
        }
    }
}

void Layer_loss::softmax(s_idx1d_ds &v_masked) {
    if (v_masked.dim1 > 0) {
        int num_masked = v_masked.dim1;
        #pragma omp parallel for
        for (int i=0; i<num_masked; i++) {
            int vid_masked = v_masked.arr[i];
            double sum_exp = 0;
            for (int j=0; j<num_cls; j++) {
                pred.arr[i+j*num_masked] = expf(feat_in.arr[vid_masked+j*feat_in.dim1]);
                sum_exp += pred.arr[i+j*num_masked];
            }
            for (int j=0; j<num_cls; j++) {
                pred.arr[i+j*num_masked] /= sum_exp;
            }
        }
    } else {
        int dim1 = feat_in.dim1;
        int dim2 = feat_in.dim2;
        #pragma omp parallel for
        for (int i=0; i<dim1; i++) {
            double sum_exp = 0;
            for (int j=0; j<dim2; j++) {
                pred.arr[i+j*dim1] = expf(feat_in.arr[i+j*dim1]);
                sum_exp += pred.arr[i+j*dim1];
            }
            for (int j=0; j<dim2; j++) {
                pred.arr[i+j*dim1] /= sum_exp;
            }
        }
    }
}


/*
 * want to mask v if in val/test mode,
 * in this case, don't need to back prop
 */
void Layer_loss::f1_score(s_idx2d_ds label, s_stat_acc &loss_acc) {
    // [accuracy]   (tp+tn)/(tp+tn+fp+fn)
    // [precision]  tp+fp>0 ? tp/(tp+fp) : 0
    // [recall]     tp+fn>0 ? tp/(tp+fn) : 0
    // [f1]         recall+precision>0 ? 2*(recall*precision)/(recall+precision) : 0
    int num_cls_mac;        // num of classes appeared
    bool *cls_flag = new bool[num_cls];
    memset(cls_flag, 0, num_cls*sizeof(bool));
    if (is_sigmoid) {
        num_cls_mac = num_cls;
        for (int i=0; i<size_subg; i++) {
            for (int j=0; j<num_cls; j++) {
                pred_bin.arr[i+j*size_subg] = pred.arr[i+j*size_subg]>0.5 ? 1 : 0;
            }
        }
    } else {
        num_cls_mac = 0;
        for (int i=0; i<size_subg; i++) {
            for (int j=0; j<num_cls; j++) {
                cls_flag[j] = (cls_flag[j] || (label.arr[i+j*size_subg]==1));
            }
        }
        for (int i=0; i<size_subg; i++) {
            int max_idx = -1;
            double max = -INFINITY;
            for (int j=0; j<num_cls; j++) {
                if (pred.arr[i+j*size_subg]>max) {
                    max = pred.arr[i+j*size_subg];
                    max_idx = j;
                }
            }
            for (int j=0; j<num_cls; j++) {
                pred_bin.arr[i+j*size_subg] = 0;
            }
            pred_bin.arr[i+max_idx*size_subg] = 1;
            cls_flag[max_idx] = true;
        }
        for (int i=0; i<num_cls; i++) {
            num_cls_mac += cls_flag[i] ? 1 : 0;
        }
    }
    s_idx2d_ds comb(size_subg,num_cls);
    // combine/encode the two matrices, so that:
    // TP -- 11
    // FN -- 10
    // FP -- 01
    // Tn -- 00
    for (int i=0; i<size_subg; i++) {
        for (int j=0; j<num_cls; j++) {
            comb.arr[i+j*size_subg] = 10*label.arr[i+j*size_subg]+pred_bin.arr[i+j*size_subg];
        }
    }
    // calc accuracy
    double accuracy_cls(0.), precision_cls(0.), recall_cls(0.), f1_accum(0.);
    t_idx tp_accum(0), fn_accum(0), fp_accum(0), tn_accum(0);
    t_idx tp_cls, fn_cls, fp_cls, tn_cls;
    for (int c=0; c<num_cls; c++)
    {
        tp_cls = 0;
        fn_cls = 0;
        fp_cls = 0;
        tn_cls = 0;
        for (int d=0; d<size_subg; d++)
        {
            if (comb.arr[d+c*comb.dim1]==11) tp_cls+=1;
            else if (comb.arr[d+c*comb.dim1]==10) fn_cls+=1;
            else if (comb.arr[d+c*comb.dim1]==1) fp_cls+=1;
            else tn_cls+=1;
        }
        tp_accum += tp_cls;
        fn_accum += fn_cls;
        fp_accum += fp_cls;
        tn_accum += tn_cls;

        accuracy_cls = (double)(tp_cls+tn_cls)/(double)(tp_cls+tn_cls+fp_cls+fn_cls);
                            //accuracy(tp_cls,tn_cls,fp_cls, fn_cls);
        precision_cls = tp_cls+fp_cls>0 ? (double)tp_cls/(double)(tp_cls+fp_cls) : 0.;
                            //precision(tp_cls,tn_cls,fp_cls,fn_cls);
        recall_cls = tp_cls+fn_cls>0 ? (double)tp_cls/(double)(tp_cls+fn_cls) : 0.;
                            //recall(tp_cls,tn_cls,fp_cls,fn_cls);
        f1_accum += recall_cls+precision_cls>0. ? 2.*(recall_cls*precision_cls)/(recall_cls+precision_cls) : 0.;
                            //f1(recall_cls,precision_cls);
    }
    double f1_mac = f1_accum/(double)num_cls_mac;
    double accuracy_mic = (double)(tp_accum+tn_accum)/(double)(tp_accum+tn_accum+fp_accum+fn_accum);
                            //accuracy(tp_accum,tn_accum,fp_accum,fn_accum);
    double precision_mic = tp_accum+fp_accum>0 ? (double)tp_accum/(double)(tp_accum+fp_accum) : 0.;
                            //precision(tp_accum,tn_accum,fp_accum,fn_accum);
    double recall_mic = tp_accum+fn_accum>0 ? (double)tp_accum/(double)(tp_accum+fn_accum) : 0.;
                            //recall(tp_accum,tn_accum,fp_accum,fn_accum);
    double f1_mic = recall_mic+precision_mic>0. ? 2.*(recall_mic*precision_mic)/(recall_mic+precision_mic) : 0.;
                            //f1(recall_mic,precision_mic);
    loss_acc.f1_mic = f1_mic;
    loss_acc.f1_mac = f1_mac;
}


void Layer_loss::update_size_subg(int size_subg_) {
    size_subg = size_subg_;
    pred.dim1 = size_subg;
    pred_bin.dim1 = size_subg;
}

void Layer_loss::forward(s_idx2d_ds label, s_stat_acc &loss_acc, s_idx1d_ds &v_masked)
{
    update_size_subg(label.dim1);
    // generate prediction in pred
    if (is_sigmoid) {sigmoid(v_masked);}
    else {softmax(v_masked);}
    // calc acc
    f1_score(label, loss_acc);
}

void Layer_loss::backward(s_idx2d_ds label, s_data2d_ds &grad_out)
{
    // grad_out = (pred - label)/batch_size
    grad_out.dim1 = size_subg;
    grad_out.dim2 = num_cls;
    #pragma omp parallel for
    for (int i=0; i<size_subg*num_cls; i++) {
        grad_out.arr[i] = pred.arr[i] - label.arr[i];
        grad_out.arr[i] /= (float)size_subg;
    }
}


