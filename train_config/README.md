### Training Configuration



#### Network:

* *dim*: hidden dimension of all layers
* *aggr*: how to aggregate the self feature and neighbor feature
* *loss*: loss function to choose (sigmoid for multi-label / softmax for single label)
* *arch*: network architecture. `1` means an order 1 layer (self feature plus 1-hop neighbor feature), and `0` means an order 0 layer (self feature only).
  * NOTE: a graph conv layer in S-GCN is equivalent to a `1-0` structure in GraphSAINT; a graph conv layer in other baselines is equivalent to a `1` layer in GraphSAINT. 
  * For the above reason, when evaluating PPI and Reddit (used in the S-GCN paper), we use `1-0-1-0`. When evaluating Flickr andYelp, we use `1-1-0` (where the last `0` is for the classifier).
  * We believe such decision on architecture gives us the fairest comparison with baselines.
* *act*: activation (I / relu / leaky\_relu), where `I` is for linear activation
* *bias*: can be `bias` or `norm` (meaning that batch norm is applied). S-GCN uses batch norm, and so GraphSAINT also uses batch norm in all configurations

#### Hyperparameters:

* *lr*: learning rate for Adam optimizer
* *sample\_coverage*: the `N` number in Section 5.3

#### Phase:

Specification of sampling parameters
