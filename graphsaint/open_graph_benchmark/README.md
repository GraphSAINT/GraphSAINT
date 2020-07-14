## Training with [Open Graph Benchmark](https://ogb.stanford.edu/)

We demonstrate GraphSAINT by tuning its performance on datasets in OGB. Training scripts in this directory are modified from the PyTorch implementation `../pytorch_version/train.py`. For each OGB graph, we run 10 times without fixing random seeds. Learning quality is measured by the `evaluator` of the corresponding dataset. 

Before training, we pre-convert the dataset into the GraphSAINT format. The converter script for the node prediction tasks is given in `../../data/open_graph_benchmark/ogbn_converter.py`. Before training, run the converter script to generate the five files required for the target dataset: 

* `adj_full.npz`
* `adj_train.npz`
* `role.json`
* `class_map.json`
* `feats.npy`

Please refer to the **How to Prepare Your Own Dataset?** section of the main README (i.e., `../../README.md`) for explanation of the five files. 


### ogbn-products

First, place the converted five data files under `../../data/open_graph_benchmark/ogbn-products/`.

Then go to the root directory of this repo (i.e., from the current directory, `cd ../../`). Run from the terminal:

```
python -m graphsaint.open_graph_benchmark.train_ogbn-products --train_config train_config/open_graph_benchmark/ogbn-products_3_e_gat.yml --data_prefix ./data/open_graph_benchmark/ogbn-products/ --gpu 0 --cpu_eval
```

You should get around 0.8027 accuracy averaged over 10 runs. Note that GraphSAINT training on ogbn-products is inductive, while most other results on the leaderboard are transductive. 
