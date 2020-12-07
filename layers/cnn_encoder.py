import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super(CNNEncoder, self).__init__()
        self.encoder = nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)

    def forward(self, token):
        pass
