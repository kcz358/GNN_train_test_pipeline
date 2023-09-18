import argparse
import datetime
from functools import partial
import json

import torch
import torch_geometric.datasets as geo_datasets
from torch_geometric.data import Data, ClusterData
from torch.utils.tensorboard import SummaryWriter

from model import GNNAutoModel



def load_config(config_file_path):
    with open(config_file_path, "r") as config_file:
        config = json.load(config_file)
    return config

def parse_arguments():
    parser = argparse.ArgumentParser(description="Argument parser for your program")

    # Dataset related arguments
    parser.add_argument("--root", type=str, required=True, help="Root directory where the dataset will be stored")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory of the model")
    parser.add_argument("--tf_log", type=str, required=True, help="Output Directory to store tensorboard path file")

    args = parser.parse_args()

    return args


def train(model, optimizer, data, criterion):
      model.train()
      optimizer.zero_grad()  # Clear gradients.
      out = model(data.x, data.edge_index)  # Perform a single forward pass.
      loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
      loss.backward()  # Derive gradients.
      optimizer.step()  # Update parameters based on gradients.
      return loss

def test(model, data, mask):
      model.eval()
      out = model(data.x, data.edge_index)
      pred = out.argmax(dim=1)  # Use the class with highest probability.
      correct = pred[mask] == data.y[mask]  # Check against ground-truth labels.
      acc = int(correct.sum()) / int(mask.sum())  # Derive ratio of correct predictions.
      return acc

def load_dataset(dataset_name):
    dataset_class = getattr(geo_datasets, dataset_name)
    return dataset_class

if __name__ == "__main__":
    torch.manual_seed(42)
    parsed_args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    config_file_path = "./config.json"  # Replace with the actual path to your config file
    config = load_config(config_file_path)
    
    # Access the parsed arguments
    dataset = config['dataset']
    dataset_name = dataset['dataset_name']
    #Popping out the dataset key and preserve other keys
    dataset.pop("dataset_name")
    
    clustering_params = config['clustering_params']
    
    root = parsed_args.root
    out_dir = parsed_args.out_dir
    tf_log = parsed_args.tf_log
    additional_dataset_parameters = dataset

    print("============================")
    # Now you can use these arguments in your program
    print("Dataset Name : ", dataset_name)
    print("Root of the dataset : ", root)
    print("Output Directory : ", out_dir)
    print("tensorboard logging Directory : ", tf_log)
    print("Additional Dataset Parameters : ", additional_dataset_parameters)
    print("Clustering parameters : ", clustering_params)
    

    
    training_params = config["training"]
    learning_rate = training_params["learning_rate"]
    epochs = training_params["epochs"]
    weight_decay = training_params["weight_decay"]
    
    model_arch = config['model']
    model_name = model_arch[0]["name"]

    print("Learning Rate:", learning_rate)
    print("Epochs:", epochs)
    print("Weight Decay:", weight_decay)
    
    print("============================")
    try:
        print("Try to load dataset {} ...".format(dataset_name))
        dataset_class = load_dataset(dataset_name)
        dataset_class = partial(dataset_class, **additional_dataset_parameters)
        dataset = dataset_class(root = root)
        data = Data(x=dataset.x, edge_index=dataset.edge_index, y=dataset.y)
        data.train_mask = dataset.train_mask
        data.val_mask = dataset.val_mask
        data.test_mask = dataset.test_mask
        data_cluster = ClusterData(data, **clustering_params)
        
        '''dataset.x = dataset.x.to(device)
        dataset.edge_index = dataset.edge_index.to(device)
        dataset.y = dataset.y.to(device)'''
    except:
        raise RuntimeError("Oops, Looks like the dataset isn't in the torchgeometric dataset class or you enter the wrong config")
    
    model = GNNAutoModel(
                         in_channels=dataset.num_features,
                         out_channels=dataset.num_classes,
                         model_arch=model_arch)
    
    print("============================")
    print("Model architecture : ")
    print(model)
    model = model.float()
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d-%H:%M:%S").replace("-", "_").replace(":", "-")

    writer = SummaryWriter(log_dir=tf_log + "/{}_{}_{}".format(model_name, dataset_name, formatted_time))
    for epoch in range(1, epochs + 1):
        epoch_loss = 0
        epoch_val_acc = 0
        epoch_test_acc = 0
        for num_cluster in range(len(data_cluster)):
            cluster = data_cluster[num_cluster]
            cluster.x = cluster.x.to(device)
            cluster.edge_index = cluster.edge_index.to(device)
            cluster.y = cluster.y.to(device)
            loss = train(model=model,
                        optimizer=optimizer,
                        data=cluster,
                        criterion=criterion)
            val_acc = test(model=model,
                        data=cluster,
                        mask=cluster.val_mask)
            test_acc = test(model=model,
                        data=cluster,
                        mask=cluster.test_mask)
            
            epoch_loss += loss
            epoch_val_acc += val_acc
            epoch_test_acc += test_acc
        epoch_loss = epoch_loss / len(data_cluster)
        epoch_val_acc = epoch_val_acc / len(data_cluster)
        epoch_test_acc = epoch_test_acc / len(data_cluster)
        print(f'Epoch: {epoch:03d}, Loss: {epoch_loss:.4f}, Val Acc: {epoch_val_acc:.4f}, Test Acc: {epoch_test_acc:.4f}')
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        writer.add_scalar("Accuracy/val", epoch_val_acc, epoch)
        writer.add_scalar("Accuracy/test", epoch_test_acc, epoch)
        if epoch % 10 == 0:
            state_dict = {
                'state_dict' : model.state_dict(),
                'epoch' : epoch,
                'val_acc' : val_acc,
                'test_acc' : test_acc
            }
            torch.save(state_dict, out_dir + "/{}_epoch".format(epoch))
    writer.close()
