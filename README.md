
# GraphSAINT: Graph <u>Sa</u>mpling Based <u>In</u>ductive Learning Me<u>t</u>hod


[Hanqing Zeng](https://sites.google.com/a/usc.edu/zengh/home)\*, [Hongkuan Zhou](https://tedzhouhk.github.io/about/)\*, [Ajitesh Srivastava](http://www-scf.usc.edu/~ajiteshs/), Rajgopal Kannan, [Viktor Prasanna](https://sites.usc.edu/prasanna/)

**Contact**


Hanqing Zeng (zengh@usc.edu), Hongkuan Zhou (hongkuaz@usc.edu)


Feel free to report bugs or tell us your suggestions!

## Overview

GraphSAINT is a general and flexible framework for training GNNs on large graphs. GraphSAINT highlights a novel minibatch method specifically optimized for data with complex relationships (i.e., graphs). The traditional way of training a GNN is: 1). Construct a GNN on the full training graph; 2). For each minibatch, pick some nodes at the output layer as the root node. Backtrack the inter-layer connections from the root node until reaching the input layer; 3). Forward and backward propagation based on the loss on the roots. The way GraphSAINT trains a GNN is: 1). For each minibatch, sample a small subgraph from the full training graph; 2). Construct a **complete** GNN on the small subgraph. No sampling is performed within GNN layers; 3). Forward and backward propagation based on the loss on the subgraph nodes.

![GraphSAINT training algorithm](./overview_diagram.png)

GraphSAINT performs "*graph sampling*" based training, whereas others perform "*layer sampling*" based training. Why does it matter to change the perspective of sampling? GraphSAINT achieves the following:

**Accuracy**: We perform simple yet effective normalization to eliminate the bias introduced by graph sampling. In addition, since any sampling process incurs information loss due to dropped neighbors, we propose light-weight graph samplers to preserve important neighbors based on topological characteristics. In fact, graph sampling can also be understood as data augmentation or training regularization (e.g., we may see the edge sampling as a minibatch version of [DropEdge](https://arxiv.org/abs/1907.10903)).

**Efficiency**: While "neighbor explosion" is a headache for many layer sampling based methods, GraphSAINT provides a clean solution to it thanks to the graph sampling philosophy. As each GNN layer is complete and unsampled, the number of neighbors keeps constant no matter how deep we go. Computation cost per minibatch reduces from exponential to linear, w.r.t. GNN depth.

**Flexibility**: Layer propagation on a minibatch subgraph of GraphSAINT is almost identical to that on the full graph. Therefore, most GNN architectures designed for the full graph can be seamlessly trained by GraphSAINT. On the other hand, some layer sampling algorithms only support limited number of GNN architectures. Take JK-net as an example: the jumping knowledge connection requires node samples in shallower layers as a superset of node samplers in the deeper layers --- minibatches of FastGCN and AS-GCN do not satisfy such condition.

**Scalability**: GraphSAINT achieves scalability w.r.t. 1). *graph size*: our subgraph size does not need to grow proportionally with the training graphs size. So even if we are dealing with a million-node graph, the subgraphs can still easily fit in the GPU memory; 2). *model size*: by resolving "neighbor explosion", training cost scales linearly with GNN width and depth; and 3). *amount of parallel resources*: graph sampling is highly scalable by trivial task parallelism. In addition, resolving "neighbor explosion" also implies dramatic reduction in communication overhead, which is critical in distributed setting (see our IEEE/IPDPS '19 or [hardware accelerator development](https://dl.acm.org/doi/abs/10.1145/3373087.3375312)).


## About This Repo

This repo contains source code of our two papers (ICLR '20 and IEEE/IPDPS '19, see the [Citation](#Citation-&-Acknowledgement) Section).

The `./graphsaint` directory contains the Python implementation of the minibatch training algorithm in ICLR '20. We provide two implementations, one in Tensorflow and the other in PyTorch. The two versions follow the same algorithm. Note that all experiments in our paper are based on the Tensorflow implementation. New experiments on open graph benchmark are based on the PyTorch version. 


The `./ipdps19_cpp` directory contains the C++ implementation of the parallel training techniques described in IEEE/IPDPS '19 (see `./ipdps19_cpp/README.md`). All the rest of this repository are for GraphSAINT in ICLR '20.

The GNN architectures supported by this repo:

|  GNN arch  |  Tensorflow  |  PyTorch  |  C++  |
| -------------: |:-------------:|:-----:|:----:|
|GraphSAGE| :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
|GAT| :heavy_check_mark: | :heavy_check_mark: | |
|JK-Net| :heavy_check_mark: | | |
| GaAN | | :heavy_check_mark: | |
|MixHop| :heavy_check_mark: | :heavy_check_mark: | |

The graph samplers supported by this repo:

|  Sampler  |  Tensorflow  |  PyTorch  |  C++  |
| -------------: |:-------------:|:-----:|:----:|
|Node| :heavy_check_mark: | :heavy_check_mark: |  |
|Edge| :heavy_check_mark: | :heavy_check_mark: | |
|RW| :heavy_check_mark: | :heavy_check_mark: | |
| MRW | :heavy_check_mark:| :heavy_check_mark: |:heavy_check_mark: | :heavy_check_mark:|
|Full graph| :heavy_check_mark: | :heavy_check_mark: | |

where
* RW: Random walk sampler
* MRW: Multi-dimensional random walk sampler
* Full graph: always returns the full training graph. Meant to be a baseline. No real "sampling" is going on.

You can add your own samplers and GNN layers easily. See the [Customization](#Customization) section.

## Results

**New**: We are testing GraphSAINT on [Open Graph Benchmark](https://ogb.stanford.edu/). Currently, we have results for the `ogbn-products` graph. Note that the `ogbn-products` accuracy on the leaderboard trained with other methods are mostly under the transductive setting. Our results are under inductive learning (which is harder).

All results in ICLR '20 can be reproduced by running the config in `./train_config/`. For example, `./train_config/table2/*.yml` stores all the config for Table 2 of our paper. `./train_config/explore/*,yml` stores all the config for deeper GNNs and various GNN architectures (GAT, JK, etc.). In addition, results related to OGB are trained by the config in `./train_config/open_graph_benchmark/*.yml`.


Test set F1-mic score summarized below.

| Sampler | Depth|  GNN | PPI | PPI (large) | Flickr | Reddit | Yelp | Amazon | ogbn-products |
|---:|:----:|:---:|:----:|:---:|:----:|:----:|:----:|:----:|:----:|
| Node | 2 | SAGE | 0.960 |  | 0.507 | 0.962 | 0.641 | 0.782 | |
| Edge | 2 | SAGE | 0.981 | | 0.510 | 0.966 | 0.653 | 0.807 | |
| RW | 2 | SAGE | 0.981 | 0.941 | 0.511 | 0.966 | 0.653 | 0.815 | |
| MRW | 2 | SAGE | 0.980 |  | 0.510 | 0.964 | 0.652 | 0.809 | |
| RW | 5 | SAGE | | 0.995 | | | | | |
| Edge | 4 | JK | | | | 0.970 | | | |
| RW | 2 | GAT | | | 0.510 | 0.967 | 0.652 | 0.815 | |
| RW | 2 | GaAN | | | 0.508 | 0.968 | 0.651 | | |
| RW | 2 | MixHop | | | | 0.967 | | | |
| Edge | 3 | GAT | | | |  | | | 0.8027






## Dependencies


* python >= 3.6.8
* tensorflow >=1.12.0  / pytorch >= 1.1.0
* cython >=0.29.2
* numpy >= 1.14.3
* scipy >= 1.1.0
* scikit-learn >= 0.19.1
* pyyaml >= 3.12
* g++ >= 5.4.0
* openmp >= 4.0


## Datasets


All datasets used in our papers are available for download:


* PPI
* PPI-large (a larger version of PPI)
* Reddit
* Flickr
* Yelp
* Amazon
* ogbn-products
* ... (more to be added)

They are available on [Google Drive link](https://drive.google.com/open?id=1zycmmDES39zVlbVCYs88JTJ1Wm5FbfLz) (alternatively, [BaiduYun link (code: f1ao)](https://pan.baidu.com/s/1SOb0SiSAXavwAcNqkttwcg)). Rename the folder to `data` at the root directory.  The directory structure should be as below:


```
GraphSAINT/
│   README.md
│   run_graphsaint.sh
│   ...
│
└───graphsaint/
│   │   globals.py
│   │   cython_sampler.pyx
│   │   ...
│   │
│   └───tensorflow_version/
│   │   │    train.py
│   │   │    model.py
│   │   │    ...
│   │
│   └───pytorch_version/
│       │    train.py
│       │    model.py
│       │    ...
│
└───data/
│   └───ppi/
│   │   │    adj_train.npz
│   │   │    adj_full.npz
│   │   │    ...
│   │
│   └───reddit/
│   │   │    ...
│   │
│   └───...
│
```


We also have a script that converts datasets from our format to GraphSAGE format. To run the script,


`python convert.py <dataset name>`


For example `python convert.py ppi` will convert dataset PPI and save new data in GraphSAGE format to `./data.ignore/ppi/`


**New**: For data conversion from the OGB format to the GraphSAINT format, please use the script `./data/open_graph_benchmark/ogbn_converter.py`. Currently, this script can handle `ogbn-products` and `ogbn-arxiv`.



## Cython Implemented Parallel Graph Sampler


We have a cython module which need compilation before training can start. Compile the module by running the following from the root directory:


`python graphsaint/setup.py build_ext --inplace`


## Training Configuration


The hyperparameters needed in training can be set via the configuration file: `./train_config/<name>.yml`.


The configuration files to reproduce the Table 2 results are packed in `./train_config/table2/`.


For detailed description of the configuration file format, please see `./train_config/README.md`


## Run Training


First of all, please compile cython samplers (see above).


We suggest looking through the available command line arguments defined in `./graphsaint/globals.py` (shared by both the Tensorflow and PyTorch versions). By properly setting the flags, you can maximize CPU utilization in the sampling step (by telling the number of available cores), select the directory to place log files, and turn on / off loggers (Tensorboard, Timeline, ...), etc.


*NOTE*: For all methods compared in the paper (GraphSAINT, GCN, GraphSAGE, FastGCN, S-GCN, AS-GCN, ClusterGCN), sampling or clustering is **only** performed during training.
To obtain the validation / test set accuracy, we run the full batch GNN on the full graph (training + validation + test nodes), and calculate F1 score only for the validation / test nodes. See also issue #11.




For simplicity of implementation, during validation / test set evaluation, we perform layer propagation using the full graph adjacency matrix. For Amazon or Yelp, this may cause memory issue for some GPUs. If an out-of-memory error occurs, please use the `--cpu_eval` flag to force the val / test set evaluation to take place on CPU (the minibatch training will still be performed on GPU). See below for other Flags.


To run the code on CPU


```
python -m graphsaint.<tensorflow/pytorch>_version.train --data_prefix ./data/<dataset_name> --train_config <path to train_config yml> --gpu -1
```


To run the code on GPU


```
python -m graphsaint.<tensorflow/pytorch>_version.train --data_prefix ./data/<dataset_name> --train_config <path to train_config yml> --gpu <GPU number>
```


For example `--gpu 0` will run on the first GPU. Also, use `--gpu <GPU number> --cpu_eval` to make GPU perform the minibatch training and CPU to perform the validation / test evaluation.


We have also implemented dual-GPU training to further speedup runtime. Simply add the flag `--dualGPU` and assign two GPUs using the `--gpu` flag. Currently this only works for GPUs supporting memory pooling and connected by NvLink.

**New**: we have prepared specific scripts to train OGB graphs. See `./graphsaint/open_graph_benchmark/` for the scripts and instructions.


## Customization

Below we describe how to customize this code base for your own research / product.

### How to Prepare Your Own Dataset?

Suppose your full graph contains N nodes. Each node has C classes, and length-F initial attribute vector. If your train/val/test split is a/b/c (i.e., a+b+c=1), then:

`adj_full.npz`: a sparse matrix in CSR format, stored as a `scipy.sparse.csr_matrix`. The shape is N by N. Non-zeros in the matrix correspond to all the edges in the full graph. It doesn't matter if the two nodes connected by an edge are training, validation or test nodes. For unweighted graph, the non-zeros are all 1.

`adj_train.npz`: a sparse matrix in CSR format, stored as a `scipy.sparse.csr_matrix`. The shape is also N by N. However, non-zeros in the matrix only correspond to edges connecting two training nodes. The graph sampler only picks nodes/edges from this `adj_train`, not `adj_full`. Therefore, neither the attribute information nor the structural information are revealed during training. Also, note that only aN rows and cols of `adj_train` contains non-zeros. See also issue #11. For unweighted graph, the non-zeros are all 1.

`role.json`: a dictionary of three keys. Key `'tr'` corresponds to the list of all training node indices. Key `va` corresponds to the list of all validation node indices. Key `te` corresponds to the list of all test node indices. Note that in the raw data, nodes may have string-type ID. You would need to re-assign numerical ID (0 to N-1) to the nodes, so that you can index into the matrices of adj, features and class labels.

`class_map.json`: a dictionary of length N. Each key is a node index, and each value is either a length C binary list (for multi-class classification) or an integer scalar (0 to C-1, for single-class classification).

`feats.npy`: a `numpy` array of shape N by F. Row i corresponds to the attribute vector of node i.


### How to Add Your Own Sampler?

All samplers are implemented as subclass of `GraphSampler` in `./graphsaint/graph_samplers.py`. There are two ways to implement your sampler subclass:

1) Implement in pure python. Overwrite the `par_sample` function of the super-class. We provide a basic example in the `NodeSamplingVanillaPython` class of `./graphsaint/graph_samplers.py`.
	* Pros: Easy to implement
	* Cons: May have slow execution speed. It is non-trivial to parallelize a pure python function.
2) Implement in cython. You need to add a subclass of the `Sampler` in `./graphsaint/cython_sampler.pyx`. In the subclass, you only need to overwrite the `__cinit__` and `sample` functions. The `sample` function defines the sequential behavior of the sampler. We automatically perform task-level parallelism by launching multiple samplers at the same time.
	* Pros: Fits in the parallel-execution framework. C++ level execution speed.
	* Cons: Hard to code

### How to Support Your Own GNN Layer?

Add a layer in `./graphsaint/<tensorflow or pytorch>_version/layers.py`. You would also need to do some minor update to `__init__` function of the `GraphSAINT` class in `./graphsaint/<tensorflow or pytorch>_version/models.py`, so that the model knows how to lookup the correct class based on the keyword in the `yml` config.

## Citation & Acknowledgement

Supported by DARPA under FA8750-17-C-0086, NSF under CCF-1919289 and OAC-1911229.

We thank Matthias Fey for providing a [reference implementation](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.GraphSAINTSampler) in the PyTorch Geometric library.

We thank the [OGB team](https://ogb.stanford.edu/) for using GraphSAINT on large scale experiments.

* ICLR 2020:

```
@inproceedings{graphsaint-iclr20,
title={{GraphSAINT}: Graph Sampling Based Inductive Learning Method},
author={Hanqing Zeng and Hongkuan Zhou and Ajitesh Srivastava and Rajgopal Kannan and Viktor Prasanna},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=BJe8pkHFwS}
}
```


* IEEE/IPDPS 2019:


```
@INPROCEEDINGS{graphsaint-ipdps19,
author={Hanqing Zeng and Hongkuan Zhou and Ajitesh Srivastava and Rajgopal Kannan and Viktor Prasanna},
booktitle={2019 IEEE International Parallel and Distributed Processing Symposium (IPDPS)},
title={Accurate, Efficient and Scalable Graph Embedding},
year={2019},
month={May},
}
