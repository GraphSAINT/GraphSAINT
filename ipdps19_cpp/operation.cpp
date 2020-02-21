#include "global.h"
#include "operation.h"
#include <omp.h>
#include <cassert>
#include <cstring>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#ifdef USE_MKL
    #include "mkl.h"
#endif


void transpose_adj(s_data2d_sp adj, t_data *arr_adj_trans) {
    t_idx* ptr_arr = (t_idx*)_malloc(adj.num_v*sizeof(t_idx));
    memcpy(ptr_arr, adj.indptr, adj.num_v*sizeof(t_idx));
    for (int i=0; i<adj.num_v; i++) {
        int v_start = adj.indptr[i];
        int v_end = adj.indptr[i+1];
        for (int j=v_start; j<v_end; j++) {
            int row_new = adj.indices[j];
            arr_adj_trans[ptr_arr[row_new]] = adj.arr[j];
            ptr_arr[row_new] ++;
        }
    }
    _free(ptr_arr);
}

void lookup_feats(s_idx1d_ds v_subg, s_data2d_ds feat_fullg, s_data2d_ds &feat_subg) {
    double t1 = omp_get_wtime();
    assert(feat_subg.dim2 == feat_fullg.dim2);
    feat_subg.dim1 = v_subg.dim1;
    #pragma omp parallel for
    for (int i=0; i<v_subg.dim1; i++) {
        for (int j=0; j<feat_subg.dim2; j++) {
            feat_subg.arr[i+j*v_subg.dim1] = feat_fullg.arr[v_subg.arr[i]+j*feat_fullg.dim1];
        }
    }
    double t2 = omp_get_wtime();
    time_ops[OP_LOOKUP] += t2-t1;
}

void lookup_labels(s_idx1d_ds v_subg, s_idx2d_ds labels_fullg, s_idx2d_ds &labels_subg) {
    double t1 = omp_get_wtime();
    assert(labels_fullg.dim2 == labels_subg.dim2);
    labels_subg.dim1 = v_subg.dim1;
    int dim1 = labels_subg.dim1;
    int dim2 = labels_subg.dim2;
    #pragma omp parallel for
    for (int i=0; i<dim1; i++) {
        int vid = v_subg.arr[i];
        for (int j=0; j<dim2; j++) {
            labels_subg.arr[i+j*dim1] = labels_fullg.arr[vid+j*labels_fullg.dim1];
        }
    }
    double t2 = omp_get_wtime();
    time_ops[OP_LOOKUP] += t2-t1;
}

void sparseMM(s_data2d_sp adj, s_data2d_ds X, s_data2d_ds &ret, int num_thread) {
    double t1 = omp_get_wtime();
    assert(adj.num_v == X.dim1);
    assert(X.dim2 == ret.dim2 && X.dim1 == ret.dim1);
    double part_feat = (float)X.dim2/(float)num_thread;
    int num_v = adj.num_v;
    #pragma omp parallel for
    for (int i=0; i<ret.dim1*ret.dim2; i++) {
        ret.arr[i] = 0.;
    }
    #pragma omp parallel for num_threads(num_thread)
    for (int i=0; i<num_thread; i++) {
        int _end_idx = (i==num_thread-1) ? ret.dim2:(int)floor((i+1)*part_feat);
        for (int row=0; row<num_v; row++) {
            for (int idx_v=adj.indptr[row]; idx_v<adj.indptr[row+1]; idx_v++) {
                int id_neigh = adj.indices[idx_v];
                assert(id_neigh <= adj.num_v-1);
                for (int idx_feat=(int)floor(i*part_feat); idx_feat<_end_idx; idx_feat++) {
                    ret.arr[idx_feat*num_v+row] += X.arr[idx_feat*num_v+id_neigh]*adj.arr[idx_v];
                }
            }
        }
    }
    double t2 = omp_get_wtime();
    time_ops[OP_SPARSE] += t2-t1;
}

inline t_data get_val(s_data2d_ds X, int r, int c, bool trans) {
    return X.arr[(trans?r:c)*X.dim1+(trans?c:r)];
}

void denseMM(s_data2d_ds X, s_data2d_ds W, s_data2d_ds &ret, bool trans1, bool trans2, bool accum) {
    double t1 = omp_get_wtime();
#ifdef USE_MKL
    if ((!trans1)&&(!trans2)) {assert(X.dim2 == W.dim1);}
    else if ((trans1)&&(!trans2)) {assert(X.dim1 == W.dim1);}
    cblas_dgemm(CblasColMajor,
                trans1?CblasTrans:CblasNoTrans,
                trans2?CblasTrans:CblasNoTrans,
                (!trans1)?X.dim1:X.dim2,
                (!trans2)?W.dim2:W.dim1,
                (!trans1)?X.dim2:X.dim1,
                1.,
                X.arr,
                X.dim1,
                W.arr,
                W.dim1,
                (int)accum,
                ret.arr,
                ret.dim1);
#else
    int dim1_out = trans1?X.dim2:X.dim1;
    int dim2_out = trans2?W.dim1:W.dim2;
    int dim_mid = trans1?X.dim1:X.dim2;
    ret.dim1 = dim1_out;
    ret.dim2 = dim2_out;
    if (!accum) {
        #pragma omp parallel for
        for (int i=0; i<dim1_out*dim2_out; i++) {
            ret.arr[i] = 0;
        }
    }
    #pragma omp parallel for
    for (int i=0; i<dim1_out; i++) {
        for (int j=0; j<dim2_out; j++) {
            for (int k=0; k<dim_mid; k++) {
                ret.arr[i+j*ret.dim1] += get_val(X,i,k,trans1)*get_val(W,k,j,trans2);
            }
        }
    }
#endif
    double t2 = omp_get_wtime();
    time_ops[OP_DENSE] += t2-t1;
}

void biasMV(s_data2d_ds &X, s_data1d_ds b) {
    double t1 = omp_get_wtime();
    // alternatively, cblas_dger
    #pragma omp parallel for
    for (int i=0; i<X.dim2; i++) {
        for (int j=0; j<X.dim1; j++) {
            X.arr[i*X.dim1+j] += b.arr[i];
        }
    }
    double t2 = omp_get_wtime();
    time_ops[OP_BIAS] += t2-t1;
}

void relu(s_data2d_ds &X, s_idx1d_ds &mask, int num_thread) {
    double t1 = omp_get_wtime();
    int load = floor(X.dim1*X.dim2/num_thread);
    std::vector<std::vector<int>> idx_zero;
    for (int i=0; i<num_thread; i++) {
        idx_zero.push_back(std::vector<int>());
    }
    #pragma omp parallel default (shared) 
    {
        int ompTid = omp_get_thread_num();
        int idx_end = (ompTid==num_thread-1) ? X.dim1*X.dim2:(ompTid+1)*load;
        for (int i=ompTid*load; i<idx_end; i++) {
            if (X.arr[i] < 0) {
                X.arr[i] = 0;
                idx_zero[ompTid].push_back(i);  // ALTERNATIVE WAY: alloc a static array of full size V x f
            }
        }
    }
    double t2 = omp_get_wtime();
    int offset = 0;
    for (auto itr=idx_zero.begin(); itr<idx_zero.end(); itr++) {
        copy(itr->begin(),itr->end(),mask.arr+offset);
        offset += itr->size();
    }
    mask.dim1 = offset;
    //double t2 = omp_get_wtime();
    time_ops[OP_RELU] += t2-t1;
}

void maskM(s_data2d_ds &X, s_idx1d_ds mask) {
    double t1 = omp_get_wtime();
    #pragma omp parallel for
    for (int i=0; i<mask.dim1; i++) {
        X.arr[mask.arr[i]] = 0;
    }
    double t2 = omp_get_wtime();
    time_ops[OP_MASK] += t2-t1;
}

void reduce_sum(s_data2d_ds X, s_data1d_ds &ret, int axis) {
    double t1 = omp_get_wtime();
    assert(axis == 0);      // right now only supports accum by axis 0
    assert(X.dim2 == ret.dim1);
    #pragma omp parallel for
    for (int i=0; i<ret.dim1; i++) {
        ret.arr[i] = 0.;
    }
    #pragma omp parallel for
    for (int i=0; i<X.dim2; i++) {
        for (int j=0; j<X.dim1; j++) {
            ret.arr[i] += X.arr[j+i*X.dim1];
        }
    }
    double t2 = omp_get_wtime();
    time_ops[OP_REDUCE] += t2-t1;
}
