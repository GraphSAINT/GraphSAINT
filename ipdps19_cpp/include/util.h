#pragma once
#include <vector>
#include <string>
#include "global.h"

void bind_to_proc(int pid);

double avg_stat_time_1d(int iter_start, int iter_end, double *time);
double avg_stat_time_2d(int iter_start, int iter_end, int layer_start, int layer_end, double **time);
void print_stat_time(s_stat_time time_acc);

/*
 * Hardcode the args for now. May switch to GNU GetOpt / Boost program_options in future
 */
void parse_args(int argc, char* argv[], char *&data, int &num_itr, int &num_thread, int &num_layer, int &size_subg, 
    int &size_frontier, int &size_hid, double &rate_learn, bool &is_sigmoid);
 
/*
 * To setup data structures related with graph dataset.
 */
void load_data(char *data, s_data2d_sp &adj_full, s_data2d_sp &adj_train,
    s_idx1d_ds &node_all, s_idx1d_ds &node_train, s_idx1d_ds &node_val, s_idx1d_ds &node_test,
    s_data2d_ds &input, s_idx2d_ds &output);

void free_data2d_sp(s_data2d_sp adj);

