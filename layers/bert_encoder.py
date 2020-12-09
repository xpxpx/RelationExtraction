import torch
import torch.nn as nn
from transformers import AutoModel


class BERTEncoder(nn.Module):
    def __init__(self, bert_file):
        super(BERTEncoder, self).__init__()
        self.encoder = AutoModel.from_pretrained(bert_file)

    def forward(self, token, attention_mask):
        return self.encoder(token, attention_mask=attention_mask)
