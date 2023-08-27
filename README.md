# GNN_train_test_pipeline
A simple GNN training and testing node classification pipeline using torchgeometric as tools

# Usage

To start training the model, you can use

```
python main.py \
        --dataset_name=Planetoid \
        --root=./datasets \
        --out_dir=./checkpoints \
        --tf_log=./tf_log \
        --message_layer=GATConv \
        --num_layers=2 \
        --hid_dim=16 \
        --additional_dataset_parameters="{'name' : 'Cora'}"\
        --additional_model_parameters="{'head' : 8}"
```

This would start to train a GNN neural network with 2 GAT conv layers and 16 hidden dimension that do node classification on the dataset Planetoid Cora.
