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

    "model" : [
        {
            "layer_id" : 1,
            "name" : "GATConv",
            "heads" : 8,
            "out_channels" : 8
        },

        {
            "layer_id" : 2,
            "name" : "GATConv",
            "heads" : 1,
            "concat" : false,
            "in_channels" : 64
        }
    ],

    "dataset" : {
        "dataset_name" : "Planetoid",
        "root" : "./datasets",
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

This would start to train the GAT model using the Planetoid dataset
