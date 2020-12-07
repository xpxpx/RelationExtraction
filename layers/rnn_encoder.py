import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTMEncoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim // 2, batch_first=True, bidirectional=True)

    def forward(self, token, length):
        packed = pack_padded_sequence(token, length, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.encoder(packed)
        encoder_out, _ = pad_packed_sequence(packed_out, batch_first=True)
        return encoder_out


class GRUEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GRUEncoder, self).__init__()
        self.encoder = nn.GRU(input_dim, hidden_dim // 2, batch_first=True, bidirectional=True)

    def forward(self, token, length):
        packed = pack_padded_sequence(token, length, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.encoder(packed)
        encoder_out, _ = pad_packed_sequence(packed_out, batch_first=True)
        return encoder_out
