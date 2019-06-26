# GraphSAINT: Graph <u>Sa</u>mpling Based <u>In</u>ductive Learning Me<u>t</u>hod

This is the open source implementation for the "GraphSAINT" paper submitted to NeurIPS 2019.

With better hyperparameter searching procedure, we keep improving our results. Now GraphSAINT performs even better in terms of both accuracy and time (compared to Table 2 in the submitted paper). **NOTE**: baseline performance remains unchanged (see `./train_config/README.md` for detailed description of the parameter searching procedure). 

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

![avatar][base64str]
[Updated Table 2 and convergence curve to be added soon]

[New results with deeper GCNs and other architectures coming soon]

## Features

As stated in the paper, GraphSAINT can be easily extended to support various graph samplers, as well as other GCN architectures. 
To add customized sampler, implement the new sampler class in `./graphsaint/cython_sampler.pyx`. 

As for the GCN architecture:

* Higher order graph convolutional layers are already supported in this codebase. Just specify the order in the configuration file (see `./train_config/README.md`, and also `./train_config/explore/reddit2_rw.yml` for an example order two GCN reaching 0.967 F1-micro). 
* We will add support for Jumping Knowledge GCN (JK-Net) soon. JK-Net adopts the neighbor sampling strategy of GraphSAGE, where neighbor explosion in deeper layers is **not** resolved. We believe the graph sampling technique of GraphSAINT can be naturally applied to the architecture of JK-Nets. 

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

[base64str]:data:image/png;base64,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


