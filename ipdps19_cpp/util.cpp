#include <cstring>
#include <string>
#include <sstream>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#ifdef USE_MKL
    #include "mkl.h"
#endif
#include "global.h"
#include "util.h"
#include <numeric>

using namespace std;

void bind_to_proc(int pid) {
    cpu_set_t new_mask;
    cpu_set_t was_mask;
    CPU_ZERO(&new_mask);
    CPU_SET((long long)pid,&new_mask);
    if (sched_getaffinity(0, sizeof(was_mask), &was_mask) == -1) { 
        printf("Error: sched_getaffinity(.., sizeof(was_mask), &was_mask)\n");
    }
    if (sched_setaffinity(0, sizeof(new_mask), &new_mask) == -1) {
        printf("Error: sched_setaffinity(.., sizeof(new_mask), &new_mask)\n");
    }
}

double avg_stat_time_1d(int iter_start, int iter_end, double *time) {
    double ret = 0.;
    for (int i=iter_start; i<iter_end; i++) ret += time[i];
    return ret/(iter_end-iter_start);
}

double avg_stat_time_2d(int iter_start, int iter_end, int layer_start, int layer_end, double **time) {
    double ret = 0.;
    for (int i=iter_start; i<iter_end; i++)
        for (int j=layer_start; j<layer_end; j++)
            ret += time[i][j];
    return ret/(iter_end-iter_start);
}

void print_stat_time(s_stat_time time_arr) {
    int num_iter_warmup = 2;
    int iter = time_arr.iter_tot;
    int layer = time_arr.layer_tot;
    printf("=============================\n");
    printf(" TIME STAT FOR %d ITERATIONS \n", iter-num_iter_warmup);
    printf("=============================\n");
    printf("\tlookup time:           %.4fs\n", avg_stat_time_1d(num_iter_warmup, iter, time_arr.time_lookup));
    for (int l=0; l<layer; l++)
        printf("\tforward sparse (%4d): %.4fs\n", l, avg_stat_time_2d(num_iter_warmup, iter, l,l+1, time_arr.time_forward_sparse));
    for (int l=0; l<layer; l++)
        printf("\tforward dense  (%4d): %.4fs\n", l, avg_stat_time_2d(num_iter_warmup, iter, l,l+1, time_arr.time_forward_dense));
    for (int l=0; l<layer-1; l++)
        printf("\tforward relu   (%4d): %.4fs\n", l, avg_stat_time_2d(num_iter_warmup, iter, l,l+1, time_arr.time_forward_relu));
    for (int l=layer-1; l<layer; l++)
        printf("\tforward norm   (%4d): %.4fs\n", l, avg_stat_time_2d(num_iter_warmup, iter, l,l+1, time_arr.time_forward_norm));
    printf("\tforward pred:          %.4fs\n", avg_stat_time_1d(num_iter_warmup, iter, time_arr.time_forward_pred));
    printf("\tbackward dense:        %.4fs\n", avg_stat_time_1d(num_iter_warmup, iter, time_arr.time_backward_dense));
    for (int l=0;l<layer;l++)
        printf("\tbackward weights(%3d): %.4fs\n", l, avg_stat_time_2d(num_iter_warmup, iter, l,l+1, time_arr.time_backward_d_weights));
    for (int l=0;l<layer-1;l++)
        printf("\tbackward feats (%4d): %.4fs\n", l, avg_stat_time_2d(num_iter_warmup, iter, l,l+1, time_arr.time_backward_d_feats));
    printf("\ttotal  iteration     : %.4fs\n", avg_stat_time_1d(num_iter_warmup, iter, time_arr.time_tot));  
}


void parse_args(int argc, char* argv[], char *&data, int &num_itr, int &num_thread, int &num_layer, int &size_subg, 
    int &size_frontier, int &size_hid, double &rate_learn, bool &is_sigmoid) {
    if (argc <= 4) {
        printf("Usage: ./train data num_itr num_thread type_loss size_hid num_layer size_subg size_frontier rate_learn\n");
        exit(1);
    }
    data = argv[1];
    num_itr = atoi(argv[2]);
    num_thread = atoi(argv[3]);
    is_sigmoid = std::string(argv[4])=="sigmoid" ? true:false;
    if (argc == 5) {
        // use default values
        num_layer = DEFAULT_NUM_LAYER;
        size_subg = DEFAULT_SIZE_SUBG;
        size_frontier = DEFAULT_SIZE_FRONTIER;
        size_hid = DEFAULT_SIZE_HID*2;
        rate_learn = DEFAULT_RATE_LEARN;
    } else {
        assert(argc == 10);
        size_hid = atoi(argv[5])*2;
        num_layer = atoi(argv[6]);
        size_subg = atoi(argv[7]);
        size_frontier = atoi(argv[8]);
        rate_learn = atof(argv[9]);
    }
}

void norm_adj(s_data2d_sp &adj) {
    t_idx num_e;
    for (t_idx i=0; i<adj.num_v; i++) {
        num_e = adj.indptr[i+1]-adj.indptr[i];
        for (int k=0; k<num_e; k++) {
            adj.arr[adj.indptr[i]+k] = 1./(double)num_e;
        }
    }
}

void load_data(char *data, s_data2d_sp &adj_full, s_data2d_sp &adj_train,
    s_idx1d_ds &node_all, s_idx1d_ds &node_train, s_idx1d_ds &node_val, s_idx1d_ds &node_test,
    s_data2d_ds &input, s_idx2d_ds &output) {

    char file_adj_train_indices[1024],file_adj_train_indptr[1024],file_adj_full_indices[1024],file_adj_full_indptr[1024];
    char file_node_test[1024],file_node_train[1024],file_node_val[1024],file_input[1024],file_output[1024],file_dims[1024];
  
    snprintf(file_dims,1024,"../data_cpp/%s/dims.bin",data);
    snprintf(file_adj_train_indices,1024,"../data_cpp/%s/adj_train_indices.bin",data);
    snprintf(file_adj_train_indptr,1024,"../data_cpp/%s/adj_train_indptr.bin",data);
    snprintf(file_adj_full_indices,1024,"../data_cpp/%s/adj_full_indices.bin",data);
    snprintf(file_adj_full_indptr,1024,"../data_cpp/%s/adj_full_indptr.bin",data);
    snprintf(file_node_train,1024,"../data_cpp/%s/node_train.bin",data);
    snprintf(file_node_test,1024,"../data_cpp/%s/node_test.bin",data);
    snprintf(file_node_val,1024,"../data_cpp/%s/node_val.bin",data);
    snprintf(file_input,1024,"../data_cpp/%s/feats_norm_col.bin",data);
    snprintf(file_output, 1024,"../data_cpp/%s/labels_col.bin",data);
    t_idx dims[11];
    std::ifstream ifs;

    ifs.open(file_dims,std::ios::binary|std::ios::in);
    ifs.read((char*)dims,sizeof(t_idx)*11);
    ifs.close();
    adj_train.indices=new t_idx[dims[0]];
    ifs.open(file_adj_train_indices,std::ios::binary|std::ios::in);
    ifs.read((char*)adj_train.indices,sizeof(t_idx)*dims[0]);
    ifs.close();
    adj_train.indptr=new t_idx[dims[1]];
    ifs.open(file_adj_train_indptr,std::ios::binary|std::ios::in);
    ifs.read((char*)adj_train.indptr,sizeof(t_idx)*dims[1]);
    ifs.close();
    adj_train.num_v = dims[1]-1;
    adj_train.num_e = dims[0];
    adj_train.arr = (t_data*)_malloc(dims[0]*sizeof(t_data));
    norm_adj(adj_train);

    adj_full.indices=new t_idx[dims[2]];
    ifs.open(file_adj_full_indices,std::ios::binary|std::ios::in);
    ifs.read((char*)adj_full.indices,sizeof(t_idx)*dims[2]);
    ifs.close();
    adj_full.indptr=new t_idx[dims[3]];
    ifs.open(file_adj_full_indptr,std::ios::binary|std::ios::in);
    ifs.read((char*)adj_full.indptr,sizeof(t_idx)*dims[3]);
    ifs.close();
    adj_full.num_v = dims[3]-1;
    adj_full.num_e = dims[2];
    adj_full.arr = (t_data*)_malloc(dims[2]*sizeof(t_data));
    norm_adj(adj_full);

    node_train.arr=new t_idx[dims[4]];
    node_train.dim1 = dims[4];
    ifs.open(file_node_train,std::ios::binary|std::ios::in);
    ifs.read((char*)node_train.arr,sizeof(t_idx)*dims[4]);
    ifs.close();

    node_test.arr=new t_idx[dims[5]];
    node_test.dim1 = dims[5];
    ifs.open(file_node_test,std::ios::binary|std::ios::in);
    ifs.read((char*)node_test.arr,sizeof(t_idx)*dims[5]);
    ifs.close();

    node_val.arr=new t_idx[dims[6]];
    node_val.dim1 = dims[6];
    ifs.open(file_node_val,std::ios::binary|std::ios::in);
    ifs.read((char*)node_val.arr,sizeof(t_idx)*dims[6]);
    ifs.close();
    
    input.arr=new t_data[dims[7]*dims[8]];
    input.dim1 = dims[7];
    input.dim2 = dims[8];
    ifs.open(file_input,std::ios::binary|std::ios::in);
    ifs.read((char*)input.arr,sizeof(t_data)*dims[7]*dims[8]);
    ifs.close();

    node_all = s_idx1d_ds(adj_full.num_v);
    assert(adj_full.num_v == input.dim1);
    for (int i=0; i<adj_full.num_v; i++) {
        node_all.arr[i] = i;
    }

    output = s_idx2d_ds(dims[9],dims[10]);
    s_data2d_ds output_float = s_data2d_ds(dims[9],dims[10]);
    ifs.open(file_output,std::ios::binary|std::ios::in);
    ifs.read((char*)output_float.arr,sizeof(t_data)*dims[9]*dims[10]);
    ifs.close();
    for (int i=0; i<output.dim1*output.dim2; i++) {
        output.arr[i] = (t_data)(output_float.arr[i]);
    }
    // TODO: free memory of output_float
}



void free_data2d_sp(s_data2d_sp adj) {
    _free(adj.indices);
    _free(adj.indptr);
    _free(adj.arr);
}

