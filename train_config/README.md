## Hyperparameter Search Procedure for ICLR '20 Experiments

* Hidden dimension for each method, model and dataset: 128, 256, 512
* Dropout: 0.0, 0.1, 0.2, 0.3
* Optimizer: Adam
* Learning rate: 0.1, 0.01, 0.001, 0.0001

(For 5-layer PPI-large, we use hidden dimension of 2048 to be consistent with the architecture used in the [ClusterGCN paper](https://arxiv.org/abs/1905.07953).)


## Training Configuration

Below we describe how to write the configuration file `./train_config/<name>.yml` to start your own training. 
You can open any `*.yml` file in `./train_config/` to better understand the below sections. 

#### Network:

* *dim*: `[int]` hidden dimension of all layers
* *aggr*: `['concat' / 'mean']` how to aggregate the self feature and neighbor features
* *loss*: `['sigmoid' / 'softmax']` loss function to choose (sigmoid for multi-label / softmax for single label)
* *arch*: `['<int>-<int>-...']` network architecture. `1` means an order 1 layer (self feature plus 1-hop neighbor feature), and `0` means an order 0 layer (self feature only). Following the convention in the literature, an L-layer GCN contains L number of layers with order *at least 1*. 
  * **NOTE**: a graph conv layer in [S-GCN](https://arxiv.org/abs/1710.10568) (a.k.a. VR-GCN) is equivalent to a `1-0` structure in GraphSAINT; a graph conv layer in other baselines is equivalent to a `1` layer in GraphSAINT. 
  * For the above reason, when evaluating PPI and Reddit (which are also evaluated in the S-GCN paper), GraphSAINT uses `1-0-1-0` architecture. When evaluating Flickr, Yelp and Amazon, GraphSAINT uses `1-1-0` (where the last `0` is for the dense layer of the node classifier).
  * Alternatively for PPI and Reddit, you can simply replace `1-0-1-0` with `1-1-0`, and the resulting GraphSAINT still has significant accuracy improvement compared with other baselines. 
* *act*: `['I' / 'relu' / 'leaky_relu']` activation function, where `I` is for linear activation. For `leaky_relu`, the current version of the code supports only the default alpha value.
* *bias*: `['bias' / 'norm']` whether to apply bias or batch norm at the end of each conv layer. S-GCN uses batch norm, and so GraphSAINT also supports S-GCN style of batch norm implementation. In general, batch norm layers do not have significant impact on accuracy or convergence. 
* *jk* (optional): `['concat' / 'max_pool']` if this field is not specified, we will not add a aggregation layer at the end of all graph conv layers. If specified, we will aggregate all the graph conv layer hidden features by concatenation or max pooling, using the architecture described in the [Jk-Net paper](https://arxiv.org/abs/1806.03536).  
* *attention* (optional): `[int]` specifies the K number of the mul-head attention defined in [GAT](https://arxiv.org/abs/1710.10903). If this line is missing, the architecture is a normal GCN architecture without attention. 

#### Hyperparameters:

* *lr*: `[float]` learning rate for Adam optimizer
* *sample\_coverage*: `[int]` the `N` number in the paper (indicates how many samples to estimate edge / node probability)
* *dropout*: `[float]` dropout value
* *eval_val_every* (optional): `[int]` run evaluation on the validation set every `n` epochs. For large datasets, validation set evaluation is a bit time-consuming, so it is not necessary to run evaluation every epoch. 

#### Phase:

The training can proceed in multiple *phases*, where in each phase we can set different sampling parameters. Note here that we abuse the notation of an "epoch". We define an "epoch" as |V|/|V_s| iterations, where |V| is the number of training nodes, and |V_s| is the average number of subgraph nodes. An iteration is a single weight update step. 

* *end*: `[int]` the termination epoch number. 

Specification of sampling parameters

Node sampler:

* *sampler*: `'node'`
* *size_subgraph*: `[int]` size of the subgraph measured in number of nodes

Edge sampler:

* *sampler*: `'edge'`
* *size_subg_edge*: `[int]` how many edges to perform independent sampling

Random walk sampler:

* *sampler*: `'rw'`
* *num_root*: `[int]` number of random walkers
* *depth*: `[int]` walk length of each walker

Multi-dimensional random walk:

* *sampler*: `'mrw'`
* *size_subgraph*: `[int]` size of the subgraph measured by number of nodes. **NOTE**: this number only specifies a node budget. This algorithm may repeatedly sample some high degree nodes, and thus the final number of subgraph nodes may be much less than this budget. 
* *size_frontier*: `[int]` size of the initial frontier nodes
* *deg_clip* (optional): `[int]` clipping the degree of a node to restrict the probability of sampling nodes with extremely high degree. **NOTE**: this clipping only restricts the sampling probability, and it does NOT change the graph topology.
