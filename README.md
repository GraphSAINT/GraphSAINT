---
output: pdf_document
---

# GraphSAINT

## Dependencies

* python >= 3.6.8
* tensorflow >=1.12.0
* cython >=0.29.2
* numpy >= 1.14.3
* scipy >= 1.1.0
* scikit-learn >= 0.19.1
* pyyaml >= 3.12
* Zython (https://github.com/ZimpleX/zython)

## Dataset

Currently available datasets:

* ppi
* reddit
* flickr
* yelp
  
They are available at [gdrive](https://drive.google.com/open?id=1zycmmDES39zVlbVCYs88JTJ1Wm5FbfLz) Rename the folder to `data` at the root directory.  The root directory should be

```
GraphSAINT
│   README.md
│   run_graphsaint.sh
│   ... 
│
└───graphsaint
│   │   supervised_models.py
│   │   supervised_train.py
│   │   ...
│   
└───data
│   └───ppi
│   │   │    adj_train.npz
│   │   │    adj_full.npz
│   │   │    ...
│   │   
│   └───reddit
│   │   │    ...
│   │
│   └───...
│
```

We also have a script that convert our dataset format to GraphSAGE format. To run the script,

`python convert.py <dataset name>`

For example `python convert.py ppi` will convert dataset PPI and will save GraphSAGE format to `/data.ignore/ppi`
  


## Cython

We have a cython module which need compile before running. Compile the module by

`python graphsaint/setup.py build_ext --inplace`

## Train Config

The hyperparameters needs in training is givin in `/train_config/<dataset><num_layer>.yml`.

For detailed description of the config, please see `/train_config/README.yml`

## Run

To run the code on cpu

`./run_graphsaint.sh <dataset_name> <path to train_config yml>`

To run the code on gpu

`./run_graphsaint.sh <dataset_name> <path to train_config yml> --gpu <GPU number>`

For example `--gpu 0` will run on the first GPU. 

## Improved Performance

We improved the GPU implementation, the new runtime are shown in the following table

![table](https://github.com/GraphSAINT/GraphSAINT/blob/master/readme_table.png)