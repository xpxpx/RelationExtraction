import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, padding_idx=None, pretrain_embedding=None):
        super(Embedding, self).__init__()
        self.emb = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        if pretrain_embedding is not None:
            self.emb.weight.data.copy_(torch.from_numpy(pretrain_embedding))

    def forward(self, token):
        return self.emb(token)
