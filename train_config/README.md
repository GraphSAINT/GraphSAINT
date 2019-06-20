## Hyperparameter Search

The configuration provided here may differ from that provided in Table 3 of the submitted Appendix. The reason is that, we have identified better hyperparameters for GraphSAINT after the NeurIPS submission. 

Unchanged hyperparameter compared with the submitted version:

* Hidden dimension of each model
* Learning rate

Updated hyperparameter compared with the submitted version:

* Dropout: we search among dropout of 0.0, 0.1, 0.2, 0.3 instead of 0.0, 0.2 as stated in the paper. 
  * Result: All baseline results keep unchanged due to such additional parameter search. GraphSAINT has identified better configuration for Reddit (using dropout of 0.1).
* Sampler parameters: for all samplers, we have evaluated additional design points based on the parameters of the specific sampler.
  * Result: For RW sampler, now walk length of 2 works the best for PPI and Flickr. 

## Training Configuration



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
