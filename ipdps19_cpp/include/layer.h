#pragma once
#include "global.h"
#include "optm.h"
#include <string>


class Layer_SAGE
{
public:
    ADAM optm;
    s_data2d_ds weight_neigh, d_weight_neigh;   // layer parameters
    s_data2d_ds weight_self, d_weight_self;     // layer parameters
    s_data1d_ds bias, d_bias;                   // layer parameters
    s_data2d_ds feat_aggr;                      // [col major] aggregated neighbor features. reuse in backprop
    s_data2d_ds feat_in;                        // [col major] input features. reuse in backprop
    s_data2d_ds grad_in;                        // [col major]
    //s_data2d_ds out_forward, out_backward;
    s_idx1d_ds mask;                            // indices used to mask gradients in backprop
    int id_layer;
    int size_subg;
    int num_thread;
    int dim_weight_in, dim_weight_out;
    bool is_act;
    Layer_SAGE();
    Layer_SAGE(int id_layer_, int size_subg_, int num_thread_, bool is_act_, int dim_weight_out_, int dim_weight_in_=-1, double lr_=0);
    void forward(s_data2d_sp subg, s_data2d_ds &feat_out);
    void backward(s_data2d_sp subg, s_data2d_sp subg_trans, s_data2d_ds &grad_out);
private:
    std::vector<s_data2d_ds*> weights, d_weights;
    std::vector<s_data1d_ds*> biases, d_biases;
    void update_size_subg(int size_subg_, s_data2d_ds &feat_out);
    bool is_order_AX_W;         // true: compute (AX)W; false: compute A(XW)
};


class Layer_dense
{
public:
    ADAM optm;
    s_data2d_ds weight, d_weight;
    s_data1d_ds bias, d_bias;
    s_data2d_ds feat_in;
    s_data2d_ds grad_in;
    s_idx1d_ds mask;
    int id_layer;
    int size_subg;
    int num_thread;
    int dim_weight_in, dim_weight_out;
    Layer_dense();
    Layer_dense(int id_layer_, int size_subg_, int num_thread_, int dim_weight_out_, int dim_weight_in_=-1, double lr_=0);
    void forward(s_data2d_ds &feat_out, int size_subg_);
    void backward(s_data2d_ds &grad_out);
private:
    std::vector<s_data2d_ds*> weights, d_weights;
    std::vector<s_data1d_ds*> biases, d_biases;
    void update_size_subg(int size_subg_, s_data2d_ds &feat_out);
};


class Layer_l2norm
{
public:
    s_data2d_ds feat_in;
    s_data2d_ds grad_in;
    int id_layer;
    int size_subg;
    int num_thread;
    int dim;
    Layer_l2norm();
    Layer_l2norm(int id_layer_, int size_subg_, int num_thread_, int dim_);
    void forward(s_data2d_ds &feat_out, int size_subg_);
    void backward(s_data2d_ds &grad_out);
private:
    void update_size_subg(int size_subg_, s_data2d_ds &feat_out);
    void l2norm(s_data2d_ds &feat_out);
    void d_l2norm(s_data2d_ds &grad_out);
};


class Layer_loss    // the final layer of the SAGE
{
public:
    s_data2d_ds feat_in;        // should be the output of last dense layer
                                // there is no grad_in for this final layer
    s_data2d_ds pred;
    s_idx2d_ds pred_bin;
    bool is_sigmoid;            // true: sigmoid;   false: softmax
    int id_layer;
    int size_subg;
    int num_thread;
    int num_cls;
    Layer_loss();
    Layer_loss(int id_layer_, int size_subg_, int num_thread_, int num_cls_, bool is_sigmoid_);
    void forward(s_idx2d_ds label, s_stat_acc &loss_acc, s_idx1d_ds &v_masked);   // should return loss + acc
    void backward(s_idx2d_ds label, s_data2d_ds &grad_out);
private:
    void update_size_subg(int size_subg_);
    void sigmoid(s_idx1d_ds &v_masked);
    void softmax(s_idx1d_ds &v_masked);
    void f1_score(s_idx2d_ds label, s_stat_acc &loss_acc);
};


struct Model {
    std::vector<Layer_SAGE> layer_SAGE;
    Layer_l2norm layer_l2norm;
    Layer_dense layer_dense;
    Layer_loss layer_loss;
};


