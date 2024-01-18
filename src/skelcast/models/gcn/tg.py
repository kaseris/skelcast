"""Typed Graph model modules"""
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphLinear(nn.Module):
    """
    N: number of nodes (joints)
    
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        #stdv = 1. / math.sqrt(self.weight.size(1))
        #self.weight.data.uniform_(-stdv, stdv)
        #if self.learn_influence:
        #    self.G.data.uniform_(-stdv, stdv)
        if len(self.weight.shape) == 3:
            self.weight.data[1:] = self.weight.data[0]
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor, g: Optional[torch.Tensor] = None) -> torch.Tensor:
        if g is None and self.learn_influence:
            g = torch.nn.functional.normalize(self.G, p=1., dim=1)
            #g = torch.softmax(self.G, dim=1)
        elif g is None:
            g = self.G
        w = self.weight[self.node_type_index]
        output = self.mm(input, w.transpose(-2, -1))
        if self.bias is not None:
            bias = self.bias[self.node_type_index]
            output += bias
        output = g.matmul(output)

        return output
    
if __name__ == '__main__':
    gl = GraphLinear(10, 10)
    gl.reset_parameters()
