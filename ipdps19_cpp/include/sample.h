/*
 * This file defines various sampling methods, including:
 *  - frontier sampling
 *  - ...
 */

#pragma once
#include "global.h"


/*
 * Return <num_thread> number of adj list for the subgraphs -- subgs
 */
void sample_frontier(s_data2d_sp adj_train, s_idx1d_ds node_train, s_data2d_sp *&subgs, s_idx1d_ds *&subgs_v, int subgraph_size, int size_frontier);

void get_eval_subg(s_data2d_sp adj_full, s_idx1d_ds node_val, s_data2d_sp &subg_eval, s_idx1d_ds &subg_v_eval);
