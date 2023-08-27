from functools import partial

import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geo_nn

class GNNAutoModel(nn.Module):
    def __init__(self, 
                 message_layer, 
                 num_layers, 
                 in_channels, 
                 out_channels, 
                 hid_dim,
                 additional_parameters = {}) -> None:
        super().__init__()
        assert num_layers >= 1
        try:
            #Creating dynamic loading class and its additional parameters
            layer_class = self.load_message_layer(message_layer=message_layer)
            layer_class = partial(layer_class, **additional_parameters)
        except:
            raise RuntimeError("Looks like the given conv layer does not exist in torch_geometric")
        
        self.layers = nn.ModuleList([])
        #First layer of the model
        self.layers.append(
            layer_class(in_channels = in_channels,
                        out_channels = hid_dim)
        )
        
        #Intermediate layer of the model
        for i in range(1, num_layers - 1):
            self.layers.append(
                layer_class(in_channels = hid_dim,
                            out_channels = hid_dim)
            )
        
        #Manually append the last layer of the model
        self.layers.append(
            layer_class(in_channels = hid_dim,
                        out_channels = out_channels)
        )
        
    
    @staticmethod 
    def load_message_layer(message_layer):
        layer_class = getattr(geo_nn, message_layer)
        return layer_class
    
    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.relu(x)
        
        return x

if __name__ == "__main__":
    model = GNNAutoModel(message_layer="SAGEConv", 
                         num_layers=5, 
                         in_channels=10, 
                         out_channels=10, 
                         hid_dim=5,
                         additional_parameters={"aggr" : "lstm"})
    print(model)
