/*
 * This file defines various building block operations of training.
 */
#pragma once
#include "global.h"


void transpose_adj(s_data2d_sp adj, t_data *arr_adj_trans);

void lookup_feats(s_idx1d_ds v_subg, s_data2d_ds feat_fullg, s_data2d_ds &feat_subg);
void lookup_labels(s_idx1d_ds v_subg, s_idx2d_ds labels_fullg, s_idx2d_ds &labels_subg);
/*
 * ret = adj X
 */
void sparseMM(s_data2d_sp adj, s_data2d_ds X, s_data2d_ds &ret, int num_thread);
/* EVERYTHING COL MAJOR BY DEFAULT
 * ret = X W
 */
void denseMM(s_data2d_ds X, s_data2d_ds W, s_data2d_ds &ret, bool trans1=false, bool trans2=false, bool accum=false);
/* [in-place]
 * X = X + b
 */
void biasMV(s_data2d_ds &X, s_data1d_ds b);
/* [in-place]
 * 1. Update in-place feat_out matrix
 * 2. Fill mask with the indices of positive values of X
 */
void relu(s_data2d_ds &X, s_idx1d_ds &mask, int num_thread);
/* [in-place]
 * Mask out elements in X to 0, based on the indices in mask
 */
void maskM(s_data2d_ds &X, s_idx1d_ds mask);
/*
 * reduce sum of X by axis 0, store in ret
 */
void reduce_sum(s_data2d_ds X, s_data1d_ds &ret, int axis=0);
/*
 * ret = sigmoid(X)
 */
void sigmoid(s_data2d_ds X, s_data2d_ds &ret);
/*
 * ret = softmax(X)
 */
void softmax(s_data2d_ds X, s_data2d_ds &ret);
