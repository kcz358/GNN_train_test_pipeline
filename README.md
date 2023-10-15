# GNN_train_test_pipeline
NTU SC4020 CZ/CE 4032 Data Mining and Analytic project 1 repo

A simple GNN training and testing node classification pipeline using torchgeometric as tools. Current testing pipeline only ensures that you can run Cora, PubMed, Citesteer under Planetoid and PPI datasets. For other datasets, modification needs to be done on the code.

# Usage

## Prepare environment
For the required packages, make sure you installed pytorch and tensorboard. You can then followed the instructions to install torch_geometric and the other packages from [this website](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)


To start training the model, you should first prepare you json config file.

An example could be like this : 

```
{
    "training": {
        "learning_rate": 0.001,
        "epochs": 50,
        "weight_decay": 0.0005,
        "label_propagation": true
    },

    "clustering_params" : {
        "num_parts" : 10
    },

    "model" : [
        {
            "layer_id" : 1,
            "name" : "ClusterGCNConv",
            "out_channels" : 64
        },

        {
            "layer_id" : 2,
            "name" : "ClusterGCNConv",
            "in_channels" : 64
        }
    ],

    "dataset" : {
        "dataset_name" : "Planetoid",
        "name" : "Cora",
        "multilabel":false,
        "multigraph":false
    }
}
```

Then you can start to train the model using

```
python main.py \
        --root=./datasets \
        --out_dir=./checkpoints \
        --tf_log=./tf_log \
```

This would start to train the ClusterGCN model with 10 graph clusters using the Planetoid dataset along with the label propagation. For other algorithm or if you don't want to perform clustering, simply set the num_parts into 1.

When you pass in the dataset config, make sure you pass in the correct multilabel and multigraph. For example, Planetoid is a not a multilabel and not a multigraph dataset, so here should be false. Another example is PPI, since it is both a multilabel and multigraph dataset, we should pass in true for both of these parameters.
