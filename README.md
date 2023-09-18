# GNN_train_test_pipeline
A simple GNN training and testing node classification pipeline using torchgeometric as tools

# Usage

To start training the model, you should first prepare you json config file.

An example could be like this : 

```
{
    "training": {
        "learning_rate": 0.001,
        "epochs": 50,
        "weight_decay": 0.0005
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
        "name" : "Cora"
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

This would start to train the ClusterGCN model with 10 graph clusters using the Planetoid dataset. For other algorithm or if you don't want to perform clustering, simply set the num_parts into 1
