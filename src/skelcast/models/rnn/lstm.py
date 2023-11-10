import torch
import torch.nn as nn


class SimpleLSTMRegressor(nn.Module):
    def __init__(self, input_size,
                 hidden_size,
                 num_layers,
                 batch_first: bool = True) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=batch_first)
        self.linear = nn.Linear(in_features=hidden_size, out_features=input_size)
    
    def forward(self, x):
        out = self.lstm(x)
        out = self.linear(x)
        return out
    