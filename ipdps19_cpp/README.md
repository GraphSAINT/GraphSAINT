# Accurate, Efficient and Scalable Graph Embedding

Hanqing Zeng*, Hongkuan Zhou*, Ajitesh Srivastava, Rajgopal Kannan, Viktor Prasanna

#### IMPORTANT: This a self-contained directory for the C++ implementation of our parallel algorithm proposed in IEEE/IPDPS '19 (journal version: JPDC 2021). If you are only interested in the GraphSAINT minibatch algorithm itself (ICLR '20), but not the parallelization techniques, please ignore this directory. 


**Contact**

Hanqing Zeng (zengh@usc.edu), Hongkuan Zhou (hongkuaz@usc.edu)

**Updates**

* 2021/02/07: Fix a bug in forward / backward prop for different dimensions. 



## Compilation

### Default: MKL-based

We rely on Intel MKL for optimized execution of dense matrix multiplication and we require compilation using Intel icc. The parallelization of sampler and subgraph feature aggregation are via OpenMP (provided with icc). You can install Intel Parallel Studio (which includes MKL, icc and OpenMP) following instructions on [this page](https://software.intel.com/en-us/get-started-with-mkl-for-linux). Students can obtain a free license for the Parallel Studio. 
Below are the versions we have tested:

* icc >= 19.0.5.281
* mkl >= 2019 Update 5

To compile, first set the environment variable `MKLROOT`, so that the `makefile` knows the correct directory to search for the library. For example, if you install Parallel Studio under `$HOME` directory, then execute the following in your shell:

```
export MKLROOT=$HOME/intel/mkl
```

Finally, run from the current directory:

```
make
```

Now you are ready to perform parallel GNN training (use GraphSAGE as the backbone architecture). 

If the above does not work, you may need to set other environment variables. See this [script](https://software.intel.com/en-us/mkl-macos-developer-guide-setting-environment-variables) provided by Intel.

### Alternative: Non-MKL based (Not recommended: training would be much slower in this case) 

If speed is not critical to you and you just want to check the functionality without going through the trouble of installing MKL, ICC, etc., you can also use the non-MKL compilation. To do this:

* Go to `global.h` and comment out line 42 (`#define USE_MKL`). 
* Replace the `makefile` with `makefile.nomkl`. 
* `make` in terminal


## Dataset

Currently available datasets:

* PPI
* Flickr
* Reddit
* Amazon


The datasets are available vis this [Google Drive link](https://drive.google.com/open?id=1hKG5Op7Ohwr1QDSNsyzx1Wc-oswe78ac). Download and move the `data_cpp` folder to the root directory (i.e., parent directory of this directory).

The datasets are stored in binary format, as explained below:

#### `dims.bin`

Eleven 32-bit integers, which specify the dimensions of various tensors stored in the other binary files. The following table lists the definition of the eleven integers, and the value of them using Reddit as an example:

|      |  Definition | e.g., Reddit |
|:----:|:-----------:|:------------:|
| 0    | length of `adj_train_indices` | 10753238 |
| 1    | length of `adj_train_indptr`  | 232966 (^)  |
| 2    | length of `adj_full_indices`  | 23213838 |
| 3    | length of `adj_full_indptr`   | 232966   |
| 4    | number of training nodes      | 151701   |
| 5    | number of validation nodes    | 55334    |
| 6    | number of test nodes          | 23699    |
| 7    | number of rows in the input feature matrix X | 232965 |
| 8    | number of cols in the input feature matrix X | 602    |
| 9    | number of rows in the output label matrix    | 232965 |
| 10   | number of cols in the output labe matrix     | 41     |



(^) **Note**: While length of `adj_train_indptr` equals number of nodes in the full graph plus 1, the training sampler will NOT obtain any node or edge information on the validation/test sets. The reasons are that:
* We remove all edges (u,v) such that u or v belongs to the validation/test sets.
* The initial frontier of the sampler are selected from the set of training nodes. 


#### `adj_train_indices.bin`

1D array of type int32. The `indices` array of the sparse `adj_train` (in CSR format).

The training adjacency matrix is of shape `|V|` by `|V|`, where `V` is the set of all nodes (training + validation + test). However, an element in `adj_train` is 1 if and only if there is an edge whose both end points are training nodes. The rest elements are all 0. 


#### `adj_train_indptr.bin`

1D array of type int32. The `indptr` array of the sparse `adj_train` (in CSR format).

#### `adj_full_indices.bin`

1D array of type int32. The `indices` array of the sparse `adj_full` (in CSR format).

The full adjacency matrix is of shape `|V|` by `|V|`, where `V` is the set of all nodes (training + validation + test). An element in `adj_full` is 1 if there is an edge in the full graph (consisting of training + validation + test nodes). 

#### `adj_full_indptr.bin`

1D array of type int32. The `indptr` array of the sparse `adj_full` (in CSR format).

#### `node_train.bin`

1D array of type int32. It stores node indices of all training nodes. 

#### `node_test.bin`

1D array of type int32. It stores node indices of all test nodes.

#### `node_val.bin`

1D array of type int32. It stores node indices of all validation nodes.

#### `feats_norm_col.bin`

2D column-major array of type float64. It stores the normalized input feature of all nodes (training + validation + test).

#### `labels_col.bin`

2D column-major array of type float64. It stores the labels of all nodes (training + validation + test).


## Run

To run the program after compilation, execute:

```
./train <dataset> <num_itr> <num_thread> <type_loss> <size_hid> <num_layer> <size_subg> <size_frontier> <rate_learn>
```

where the first 4 arguments are mandatory. If the other arguments are not provided, the program will use the default value set in `./include/global.h`. 

* Our C++ training by default uses double precision floating point numbers. You can change it by modifying `t_data` in `./include/global.h`. Note that Tensorflow by default may use `float32`, so be sure to make the data type consistent when comparing training speed with Tensorflow. 
* You can set `<num_thread>` to be the number of physical cores in your system. 
* `<type_loss>` can be either `sigmoid` or `softmax`, and can affect accuracy significantly if not set correctly. 
* `<num_itr>` specifies number of iterations (NOT number of epochs). We define *one* iteration as a forward+backward pass using *one* subgraph.

Examples:

On Reddit:

```
./train reddit 100 40 softmax
```

On PPI:

```
./train ppi 200 40 sigmoid
```

**NOTE**: to get high accuracy, it would be necessary to tweak other hyper-parameters such as learning rate, sample size, dropout (to be supported soon), etc. 
If accuracy is critical, we recommend you to use the better training algorithm as implemented in the `../graphsaint` directory (see `../README.md` for details). Alternatively, you can also go to [this repo](https://github.com/ZimpleX/gcn-ipdps19) for a Tensorflow implementation of this C++ training algorithm (you may be able to configure hyper-parameters more easily using Tensorflow than using pure C++). 

## Extensions

To train on your own data, you can use our script `convert.py` to convert from the format used in GraphSAINT (python+Tensorflow) to the format used by this C++ version.

```
python convert.py <dataset name>
```

**Citation**


```
@INPROCEEDINGS{graphsaint-ipdps19,
author={Hanqing Zeng and Hongkuan Zhou and Ajitesh Srivastava and Rajgopal Kannan and Viktor Prasanna},
booktitle={2019 IEEE International Parallel and Distributed Processing Symposium (IPDPS)},
title={Accurate, Efficient and Scalable Graph Embedding},
year={2019},
month={May},
doi={10.1109/IPDPS.2019.00056}, 
}
```

```
@article{graphsaint-jpdc21,
title = {Accurate, efficient and scalable training of Graph Neural Networks},
journal = {Journal of Parallel and Distributed Computing},
volume = {147},
pages = {166-183},
year = {2021},
issn = {0743-7315},
doi = {https://doi.org/10.1016/j.jpdc.2020.08.011},
url = {https://www.sciencedirect.com/science/article/pii/S0743731520303579},
author = {Hanqing Zeng and Hongkuan Zhou and Ajitesh Srivastava and Rajgopal Kannan and Viktor Prasanna},
}
```

**TODO**

* The current C++ code implements the algorithm described in our [IEEE/IPDPS](https://ieeexplore.ieee.org/document/8820993) paper, which is a 'premature' version of our [GraphSAINT](https://openreview.net/forum?id=BJe8pkHFwS) algorithm. The major differences are that this C++ version does not perform normalization, and currently only supports Frontier (or MRW) sampling. We will update our code to be consistent with the GraphSAINT version. 
* The current C++ version does not support dropout yet. We should be able to support dropout easily given the current code framework. 
* We will modify the C++ version so that it shares the same interface with the python+Tensorflow version. i.e., users should be able to use the same `yml` file to configure GNN architecture / hyper-parameters. Conversion of data formats should take place internally during runtime. 
