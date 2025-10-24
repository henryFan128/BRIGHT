import torch
import torch.nn as nn

class Residual(nn.Module):
    """Residual connection module"""
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module

    def forward(self, x, *args, **kwargs):
        """
        Apply residual connection to the output of the module.
        If the input x is a dictionary (for heterogeneous graphs), apply residuals to each value (tensor) in the dictionary.
        """
        output = self.module(x, *args, **kwargs)
        
        if isinstance(x, (dict, nn.ParameterDict)):
            # Apply residual connection to each node type in the dictionary
            return {node_type: x[node_type] + output[node_type] for node_type in x.keys()}
        else:
            # Original residual connection
            return x + output