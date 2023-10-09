import argparse
import datetime
from functools import partial
import json

import torch
import torch.nn.functional as F
import torch_geometric.datasets as geo_datasets
from torch_geometric.data import Data
from torch_geometric.loader import ClusterData
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


def train(model, optimizer, data, criterion, lpa=None):
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    if lpa is not None:
        out = model(data.x + lpa, data.edge_index)
    else:
        out = model(data.x, data.edge_index)  # Perform a single forward pass.
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss

def test(model, data, mask, lpa=None):
    model.eval()
    if lpa is not None:
        out = model(data.x + lpa, data.edge_index)
    else:
        out = model(data.x, data.edge_index)
    if not multilabel:
        pred = out.argmax(dim=1)  # Use the class with highest probability.
    else:
        pred = out >= 0.5
    correct = pred[mask] == data.y[mask]  # Check against ground-truth labels.
    acc = int(correct.sum()) / int(mask.sum())  # Derive ratio of correct predictions.
    return acc

def load_dataset(dataset_name):
    dataset_class = getattr(geo_datasets, dataset_name)
    return dataset_class

def dataset_to_device(dataset, device):
    dataset.x = dataset.x.to(device)
    dataset.y = dataset.y.to(device)
    dataset.edge_index = dataset.edge_index.to(device)
    
def lpa_step(y):
    #LPA for training 
    if not multigraph:
        lpa = F.one_hot(y)
        lpa = torch.tensor(lpa, dtype=torch.float32).to(device)
    else:
        lpa = torch.tensor(y, dtype=torch.float32)
    
    lpa = y_mapping(lpa)
    return lpa

if __name__ == "__main__":
    torch.manual_seed(42)
    parsed_args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    config_file_path = "./config.json"  # Replace with the actual path to your config file
    config = load_config(config_file_path)
    
    # Access the parsed arguments
    dataset = config['dataset']
    dataset_name = dataset['dataset_name']
    # Whether we are performing a multilabel classification task
    multilabel = dataset['multilabel']
    # Whether we are performing transductive learning or inductive learning
    multigraph = dataset['multigraph']
    #Popping out the dataset key and preserve other keys
    dataset.pop("dataset_name")
    dataset.pop("multilabel")
    dataset.pop("multigraph")
    
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
    label_propagation = training_params["label_propagation"]
    
    model_arch = config['model']
    model_name = model_arch[0]["name"]

    print("Learning Rate:", learning_rate)
    print("Epochs:", epochs)
    print("Weight Decay:", weight_decay)
    
    print("============================")
    try:
        print("Try to load dataset {} ...".format(dataset_name))
        dataset_class = load_dataset(dataset_name)
        if multigraph:
            print(f"Transductive Learning dataset : {dataset_name}")
            print("We have multiple graph on different dataset")
            dataset_class = partial(dataset_class, **additional_dataset_parameters)
            # Loading split = train dataset
            dataset_train = dataset_class(root = root, split = "train")
            num_features = dataset_train.num_features
            num_classes = dataset_train.num_classes
            data_train = Data(x=dataset_train.x, edge_index=dataset_train.edge_index, y=dataset_train.y)
            # Here is simple to maintain the pipeline structure
            # Every x in the graph are training node, same idea applies to val and test
            data_train.train_mask = torch.ones(size=[data_train.x.shape[0]], dtype=bool)
            data_cluster = ClusterData(data_train, **clustering_params)
            
            # Loading split = val dataset
            dataset_val= dataset_class(root = root, split = "val")
            data_val = Data(x=dataset_val.x, edge_index=dataset_val.edge_index, y=dataset_val.y)
            data_val.val_mask = torch.ones(size=[data_val.x.shape[0]], dtype=bool)
            dataset_to_device(data_val, device)
            
            # Loading split = test dataset
            dataset_test= dataset_class(root = root, split = "test")
            data_test = Data(x=dataset_test.x, edge_index=dataset_test.edge_index, y=dataset_test.y)
            data_test.test_mask = torch.ones(size=[data_test.x.shape[0]], dtype=bool)
            dataset_to_device(data_test, device)
        else:
            print(f"Inductive Learning dataset : {dataset_name}")
            print("We have only one graph for train and test")
            dataset_class = partial(dataset_class, **additional_dataset_parameters)
            dataset = dataset_class(root = root)
            num_features = dataset.num_features
            num_classes = dataset.num_classes
            data = Data(x=dataset.x, edge_index=dataset.edge_index, y=dataset.y)
            data.train_mask = dataset.train_mask
            data.val_mask = dataset.val_mask
            data.test_mask = dataset.test_mask
            data_cluster = ClusterData(data, **clustering_params)
            data_val = data
            data_test = data
            dataset_to_device(data_val, device)
            dataset_to_device(data_test, device)
        
        '''dataset.x = dataset.x.to(device)
        dataset.edge_index = dataset.edge_index.to(device)
        dataset.y = dataset.y.to(device)'''
    except:
        raise RuntimeError("Oops, Looks like the dataset isn't in the torchgeometric dataset class or you enter the wrong config")
    
    model = GNNAutoModel(
                         in_channels=num_features,
                         out_channels=num_classes,
                         model_arch=model_arch)
    
    print("============================")
    print("Model architecture : ")
    print(model)
    model = model.float()
    model = model.to(device)
    
    if label_propagation:
        y_mapping = torch.nn.Linear(in_features=num_classes, out_features=num_features).to(device)
        parameters = list(model.parameters()) + list(y_mapping.parameters())
    else:
        parameters = list(model.parameters())
    
    optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d-%H:%M:%S").replace("-", "_").replace(":", "-")
    LPA = "LPA" if label_propagation else ""

    writer = SummaryWriter(log_dir=tf_log + "/{}{}_{}_{}".format(model_name, LPA ,dataset_name, formatted_time))
    test_accuracies = []
    val_accuracies = []
    for epoch in range(1, epochs + 1):
        epoch_loss = 0
        epoch_val_acc = 0
        epoch_test_acc = 0
        for num_cluster in range(len(data_cluster)):
            cluster = data_cluster[num_cluster]
            dataset_to_device(cluster, device)
            if label_propagation:
                #LPA for training 
                lpa_train = lpa_step(cluster.y)
                #LPA for val
                lpa_val = lpa_step(data_val.y)
                #LPA for test
                lpa_test = lpa_step(data_test.y)
            else:
                lpa_train = None
                lpa_val = None
                lpa_test = None
                
                
            loss = train(model=model,
                        optimizer=optimizer,
                        data=cluster,
                        criterion=criterion,
                        lpa=lpa_train
                        )
            val_acc = test(model=model,
                        data=data_val,
                        mask=data_val.val_mask,
                        lpa=lpa_val
                        )
            test_acc = test(model=model,
                        data=data_test,
                        mask=data_test.test_mask,
                        lpa=lpa_test
                        )
            
            epoch_loss += loss
            epoch_val_acc += val_acc
            epoch_test_acc += test_acc
        epoch_loss = epoch_loss / len(data_cluster)
        epoch_val_acc = epoch_val_acc / len(data_cluster)
        epoch_test_acc = epoch_test_acc / len(data_cluster)
        val_accuracies.append(epoch_val_acc)
        test_accuracies.append(epoch_test_acc)
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
    print(f"Best Epoch : {test_accuracies.index(max(test_accuracies)) + 1}")
    print(f"Testing Accuracies result : {max(test_accuracies)}")