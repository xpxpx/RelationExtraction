import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super(CNNEncoder, self).__init__()
        self.encoder = nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.encoder.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, token):
        encoder_output = torch.relu(self.encoder(token.transpose(1, 2)))
        pool_output, _ = torch.max(encoder_output, dim=2)
        return pool_output
