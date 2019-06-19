network:
  - <dim>-<order>-<norm/bias>-<act>-<aggr>
  - <dim>-<order>-<norm/bias>-<act>-<aggr>
  ...

  ########################################
  # one line specifies arch of one layer #
  ########################################
  # <dim> dimension before concat
  # <order> order of the GCN, 0 for MLP, currently we support order = 0,1,2.
  ############
  ### NOTE ###: an order-1 layer followed by an order-0 layer is equivalent to a single layer in S-GCN (Chen, ICML'18). 
  ############
  # <norm/bias> n for batch-norm; b for bias.
  # <act> activations, relu for relu; lin for linear activation (f(x)=x).
  # <aggr> concat for concatenation of order i features; add for addition of order i features.
  
params:
  - lr: <lr>                      # learning rate for ADAM optimizer. 
    weight_decay: <weight_decay>  # weight decay as specified in GraphSAGE code. 
    norm_weight: <0/1>            # whether to apply normalization of the loss function (Sec 3.3).
    norm_adj: <rw/sym>            # how to normalize the adj matrix: rw for random walk based normalization; sym for symmetric normalization.
    model: <gs_mean/gsaint>       # gs_mean for graphsage_mean model of the GCN layer.
    q_threshold: <q_threshold>    # estimate the distribution of subgraph based on <q_threshold>*<fullgraph_size>/<subgraph_size> number of sampled subgraphs.
    q_offset: <q_offset>          # add <q_offset> to adjust the probability distribution estimated by the pre-processing sampling. 
    batch_norm: <tf.nn/tf.layers> # we provide two options to implement batch norm for GCN layers. the tf.nn based implementation is inspired by the code of S-GCN, while tf.layer is more often seen in other deep learning models.
    skip: <noskip/x-y>            # skip connection as described in ResNet design: noskip for no skip across the layers; x-y for the output of layer x added to the input of layer y.
phase:                            # Different phases can have different graph sampling algorithms.
  - end: <end>                    # end epoch of this phase.
    dropout: <dropout>
    sampler: <frontier/rw/khop/edge>
    # frontier:
    size_subgraph: <size_subgraph>
    size_frontier: <size_frontier>
    order: <order>
    max_deg: <max_deg>            # clip the degree to max_deg
    # rw:
    num_root: <num_root>
    depth: <depth>
    # khop:
    size_subgraph: <size_subgraph>
    order: <order>
    # edge:
    size_subgraph: <size_subgraph>
  ...
    
