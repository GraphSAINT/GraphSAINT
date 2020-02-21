/* Naming convension:
 *  - t_* means data type (after typedef)
 *  - s_* means struct (after typedef)
 *  - *_sp means sparse
 *  - *_ds means dense
 *  - *idx* means data struct related with indices (e.g., adj matrix)
 *  - *data* means data struct related with data (e.g., feat matrix)
 */

#pragma once


#define SAMPLE_FRONTIER 0

#define DEFAULT_NUM_LAYER 2
#define DEFAULT_SIZE_SUBG 9000
#define DEFAULT_SIZE_FRONTIER 3000
#define DEFAULT_METHOD_SAMPLE SAMPLE_FRONTIER
#define DEFAULT_SIZE_HID 128
#define DEFAULT_RATE_LEARN 0.05
#define DEFAULT_IS_SIGMOID false
#define ETA 1.5                                  // length factor of DB in sampling
#define SAMPLE_CLIP 3000                          // clip degree in sampling
#define EVAL_INTERVAL 1000

#define ADAM_LR 0.05
#define ADAM_BETA1 0.9
#define ADAM_BETA2 0.999
#define ADAM_EPSILON 0.00000001  

#define OP_DENSE 'a'
#define OP_SPARSE 'b'
#define OP_RELU 'c'
#define OP_NORM 'd'
#define OP_LOOKUP 'e'
#define OP_BIAS 'f'
#define OP_MASK 'g'
#define OP_REDUCE 'h'
#define OP_SIGMOID 'i'
#define OP_SOFTMAX 'j'

#define USE_MKL
#include <cstddef>
#include <cstdlib>
#ifdef USE_MKL
#include "mkl.h"
inline void* _malloc(size_t size) {return mkl_malloc(size,64);}
inline void _free(void* ptr) {return mkl_free(ptr);}
#else
inline void* _malloc(size_t size) {return malloc(size);}
inline void _free(void* ptr) {return free(ptr);}
#endif

#include "string.h"
#include <map>


/**** model/training params ****/
extern int num_layer;
extern int num_itr;
extern int num_thread;
extern int size_subg;
extern int size_frontier;
extern int method_sample;
extern int dim_hid;
extern double lr;
extern bool is_sigmoid;

typedef double t_data;      // use this type to denote all the values (e.g., weights, feature values, ...)
typedef int t_idx;          // use this type to denote all the indices (e.g., index of V, index of E, ...)


/**** data structures ****/
struct s_idx1d_ds {
    s_idx1d_ds() : dim1(0), arr(NULL) {};
    s_idx1d_ds(t_idx dim1_) : dim1(dim1_) {
            arr = (t_idx*)_malloc(dim1*sizeof(t_idx));
            memset(arr, 0, dim1*sizeof(t_idx));};
    t_idx dim1;
    t_idx *arr;
};

struct s_data1d_ds {
    s_data1d_ds() : dim1(0), arr(NULL) {};
    s_data1d_ds(t_idx dim1_) : dim1(dim1_) {
            arr = (t_data*)_malloc(dim1*sizeof(t_data));
            memset(arr, 0, dim1*sizeof(t_data));};
    t_idx dim1;
    t_data *arr;
};

// CSR
struct s_idx2d_sp {
    s_idx2d_sp() : num_v(0), num_e(0), indptr(NULL), indices(NULL), arr(NULL) {};
    s_idx2d_sp(t_idx num_v_, t_idx num_e_) : num_v(num_v_), num_e(num_e_) {
            indptr = (t_idx*)_malloc((num_v+1)*sizeof(t_idx));
            indices = (t_idx*)_malloc(num_e*sizeof(t_idx));
            arr = (t_idx*)_malloc(num_e*sizeof(t_idx));
            memset(indptr, 0, (num_v+1)*sizeof(t_idx));
            memset(indices, 0, num_e*sizeof(t_idx));
            memset(arr, 0, num_e*sizeof(t_idx));};
    t_idx num_v;    // length of indptr = num_v + 1
    t_idx num_e;
    t_idx *indptr;
    t_idx *indices;
    t_idx *arr;
};

// CSR
struct s_data2d_sp {
    s_data2d_sp() : num_v(0), num_e(0), indptr(NULL), indices(NULL), arr(NULL) {};
    s_data2d_sp(t_idx num_v_, t_idx num_e_) : num_v(num_v_), num_e(num_e_) {
            indptr = (t_idx*)_malloc((num_v+1)*sizeof(t_idx));
            indices = (t_idx*)_malloc(num_e*sizeof(t_idx));
            arr = (t_data*)_malloc(num_e*sizeof(t_data));
            memset(indptr, 0, (num_v+1)*sizeof(t_idx));
            memset(indices, 0, num_e*sizeof(t_idx));
            memset(arr, 0, num_e*sizeof(t_data));};
    t_idx num_v;    // length of indptr = num_v + 1
    t_idx num_e;
    t_idx *indptr;
    t_idx *indices;
    t_data *arr;
};

struct s_idx2d_ds {
    s_idx2d_ds() : dim1(0), dim2(0), arr(NULL) {};
    s_idx2d_ds(t_idx dim1_, t_idx dim2_) : dim1(dim1_), dim2(dim2_) {
            arr = (t_idx*)_malloc(dim1*dim2*sizeof(t_idx));
            memset(arr, 0, dim1*dim2*sizeof(t_idx));};
    t_idx dim1;
    t_idx dim2;
    t_idx *arr;    // 1D of dim1 x dim2
};

struct s_data2d_ds {
    s_data2d_ds() : dim1(0), dim2(0), arr(NULL) {};
    s_data2d_ds(t_idx dim1_, t_idx dim2_) : dim1(dim1_), dim2(dim2_) {
            arr = (t_data*)_malloc(dim1*dim2*sizeof(t_data));
            memset(arr, 0, dim1*dim2*sizeof(t_data));};
    t_idx dim1;
    t_idx dim2;
    t_data *arr;     // 1D of dim1 x dim2
};

struct s_stat_acc {
    //s_stat_acc() : f1_mac(-1.), f1_mic(-1.) {};
    float f1_mac;
    float f1_mic;
    double loss;
};

typedef struct stat_time {
    int iter_tot;
    int layer_tot;
    double *time_lookup;
    double **time_forward_sparse;
    double **time_forward_dense;
    double **time_forward_relu;
    double **time_forward_norm;
    double *time_forward_pred;
    double **time_backward_sparse;
    //double **time_backward_dense;
    double **time_backward_relu;
    double **time_backward_norm;
    double *time_backward_pred;
    double *time_tot;

    double *time_backward_dense;
    double **time_backward_d_weights;
    double **time_backward_d_feats;
} s_stat_time;

/**** shared matrices ****/

// structs related with dataset itself
extern s_idx2d_sp adj_full;
extern s_idx2d_sp adj_train;
extern s_idx1d_ds node_train;
extern s_idx1d_ds node_val;
extern s_idx1d_ds node_test;
extern s_idx1d_ds node_all;
extern s_data2d_ds input;       // input is input features for all V
extern s_idx2d_ds label_true;

extern s_stat_time time_arr;        // global var, to be updated within each ops inside operation.cpp
extern std::map<char,double> time_ops;
