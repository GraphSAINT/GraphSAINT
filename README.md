# GraphSAINT: Graph <u>Sa</u>mpling Based <u>In</u>ductive Learning Me<u>t</u>hod

This is the open source implementation for the "GraphSAINT" paper submitted to NeurIPS 2019.

With better hyperparameter searching procedure, we keep improving our results. Now GraphSAINT performs even better in terms of both accuracy and time (compared to Table 2 in the submitted paper). **NOTE**: baseline performance mostly remains unchanged (exception: FastGCN on Flickr has 0.01 higher test accuracy --- from 0.503 to 0.504). See `./train_config/README.md` for detailed description of the updated parameter searching procedure. 

**Highlight** of GraphSAINT new results (2 layer GCN models):

* Reddit
  * `./train_config/neurips/reddit2_rw.yml`: reaching 0.966 (from previously 0.964) F1-micro score, with the same training time
* Yelp
  * `./train_config/neurips/yelp2_e.yml`: reaching 0.654 (from previously 0.642) F1-micro score, with 2x training time; reaching 0.648 (from previously 0.642) F1-micro score, with 1x training time.
  * `./train_config/neurips/yelp2_rw.yml`: reaching 0.654 (from previously 0.640) F1-micro score, with 4x training time; reaching 0.648 (from previously 0.640) F1-micro score, with the same training time. 
* Flickr
  * `./train_config/neurips/flickr2_rw.yml`: reaching 0.513 (from previously 0.509) F1-micro score, with 0.75x training time
* PPI
  * `./train_config/neurips/ppi2_rw.yml`: reaching 0.982 (from previously 0.973) F1-micro score, with 4x training time; reaching 0.974 (from previously 0.973) F1-micro score, with 1x training time. 

2 layer convergence (Validation F1-Micro w.r.t. Training time) plot

![Alt text](converge.png)

For results using deeper GCNs and other layer architectures, please see below. 

## Features

As stated in the paper, GraphSAINT can be easily extended to support various graph samplers, as well as other GCN architectures. 
To add customized sampler, implement the new sampler class in `./graphsaint/cython_sampler.pyx`. 

As for the GCN architecture:

* Higher order graph convolutional layers are supported. Just specify the order in the configuration file (see `./train_config/README.md`, and also `./train_config/explore/reddit2_rw.yml` for an example order two GCN reaching 0.967 F1-micro). 
* Jumping Knowledge GCN (JK-Net) is supported. The JK-Net in the [original paper](https://arxiv.org/abs/1806.03536) adopts the neighbor sampling strategy of GraphSAGE, where neighbor explosion in deeper layers is **not** resolved. Here in this codebase, we demonstrate that graph sampling based minibatch of GraphSAINT can be applied to JK-Net architecture to improve training scalability w.r.t. GCN depth. Check out `./train_config/explore/reddit4_jk_rw.yml` for a 4-layer JK-Net achieving **0.968** F1-Micro on Reddit. The training time (under random walk sampler) is under 40 seconds!

## Dependencies

* python >= 3.6.8
* tensorflow >=1.12.0
* cython >=0.29.2
* numpy >= 1.14.3
* scipy >= 1.1.0
* scikit-learn >= 0.19.1
* pyyaml >= 3.12
* g++ >= 5.4.0
* openmp >= 4.0

## Dataset

Currently available datasets:

* PPI
* Reddit
* Flickr
* Yelp
  
They are available via this [Google Drive link](https://drive.google.com/open?id=1zycmmDES39zVlbVCYs88JTJ1Wm5FbfLz). Rename the folder to `data` at the root directory.  The directory structure should be as below:

```
GraphSAINT/
│   README.md
│   run_graphsaint.sh
│   ... 
│
└───graphsaint/
│   │   models.py
│   │   train.py
│   │   ...
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
  


## Cython

We have a cython module which need compilation before training can start. Compile the module by running the following from the root directory:

`python graphsaint/setup.py build_ext --inplace`

## Training Configuration

The hyperparameters needed in training can be set via the configuration file: `./train_config/<name>.yml`.

The configuration files to reproduce the Table 2 results are packed in `./train_config/neurips/` (some configuration files now produces even better results compared with Table 2).

For detailed description of the configuration file format, please see `./train_config/README.md`

## Run Training

We suggest looking through the available tensorflow command line flags defined in `./graphsaint/globals.py`. By properly setting the flags, you can maximize CPU utilization in the sampling step (by telling the number of available cores), and turn on / off Tensorboard, etc. 

To run the code on cpu

`./run_graphsaint.sh <dataset_name> <path to train_config yml>`

To run the code on gpu

`./run_graphsaint.sh <dataset_name> <path to train_config yml> --gpu <GPU number>`

For example `--gpu 0` will run on the first GPU. 

