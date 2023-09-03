from functools import partial

import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geo_nn

class GNNAutoModel(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 model_arch) -> None:
        super().__init__()
        assert len(model_arch) >= 1
        
        self.layers = nn.ModuleList([])
        for layer in model_arch:
            try:
                #Creating dynamic loading class and its additional parameters
                layer_class = self.load_message_layer(message_layer=layer["name"])
                
                #The name and the layer id can be reserved in the file
                #I choose to pop out because I think it would be better to read out the config
                layer.pop("name")
                
                #If it is the first or the last layer of the model
                #We fix the in channels or the out channels of it
                if(layer['layer_id'] == 1):
                    layer['in_channels'] = in_channels
                elif(layer['layer_id'] == len(model_arch)):
                    layer['out_channels'] = out_channels
                
                layer.pop("layer_id")
                layer_class = partial(layer_class, **layer)
                #print(layer)
                
                self.layers.append(layer_class())
            except:
                raise RuntimeError("Looks like the given conv layer config is not correct")
        
    
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
    import json
    with open("./config.json", "r") as config_file:
        config = json.load(config_file)
    model = GNNAutoModel(
                         in_channels=10, 
                         out_channels=10, 
                         model_arch=config['model'])
    print(model)
