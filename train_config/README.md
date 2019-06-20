## Hyperparameter Search

The configuration provided here may differ from that provided in Table 3 of the submitted Appendix. The reason is that, we have identified better hyperparameters for GraphSAINT after the NeurIPS submission. 

Unchanged hyperparameter compared with the submitted version:

* Hidden dimension of each model
* Learning rate

Updated hyperparameter compared with the submitted version:

* Dropout: we search among dropout of 0.0, 0.1, 0.2, 0.3 instead of 0.0, 0.2 as stated in the paper. 
  * Result: All baseline results keep unchanged with such additional parameter search. GraphSAINT has identified better configuration for Reddit and Yelp (using dropout of 0.1). See the main `README` and the configuration `./train_config/neurips/reddit2_rw.yml`, `./train_config/neurips/yelp2_mrw.yml`.
* Sampler parameters: for all samplers, we have evaluated additional design points based on the parameters of the specific sampler.
  * Result: For RW sampler, now walk length of 2 (instead of 4) works the best for PPI and Flickr. 
* Training phase: in the Appendix, we mentioned that for PPI, we used smaller subgraphs to "warm-up" training. Now to simplify the hyperparameter searching procedure, we have removed these initial phases. Therefore, every training now uses a single phase, with the same subgraph size throughout all training iterations.

## Training Configuration

Below we describe how to write the configuration file `./train_config/<name>.yml` to start your own training. 
You can open any `*.yml` file in `./train_config/neurips/` to better understand the below sections. 

#### Network:

* *dim*: `[int]` hidden dimension of all layers
* *aggr*: `['concat' / 'mean']` how to aggregate the self feature and neighbor features
* *loss*: `['sigmoid' / 'softmax']` loss function to choose (sigmoid for multi-label / softmax for single label)
* *arch*: `['<int>-<int>-...']` network architecture. `1` means an order 1 layer (self feature plus 1-hop neighbor feature), and `0` means an order 0 layer (self feature only).
  * NOTE: a graph conv layer in S-GCN is equivalent to a `1-0` structure in GraphSAINT; a graph conv layer in other baselines is equivalent to a `1` layer in GraphSAINT. 
  * For the above reason, when evaluating PPI and Reddit (which are evaluated in the S-GCN paper), GraphSAINT uses `1-0-1-0` architecture. When evaluating Flickr and Yelp, GraphSAINT uses `1-1-0` (where the last `0` is stands for dense layer for the classifier).
  * We believe such design choice on architecture gives us the fairest comparison with baselines.
* *act*: `['I' / 'relu' / 'leaky_relu']` activation function, where `I` is for linear activation. For leaky_relu, the current version of the code supports only the default alpha value.
* *bias*: `['bias' / 'norm']` whether to apply bias or batch norm at the end of each conv layer. S-GCN uses batch norm, and so GraphSAINT also uses batch norm in all `./train_config/neurips/` configurations. 

#### Hyperparameters:

* *lr*: learning rate for Adam optimizer
* *sample\_coverage*: the `N` number in Section 5.3

#### Phase:

Specification of sampling parameters
