import torch
import torch.nn as nn
from layers.bert_encoder import BERTEncoder


class BERT(nn.Module):
    def __init__(self, config):
        super(BERT, self).__init__()
        self.encoder = BERTEncoder(config.bert_file)
        self.linear = nn.Linear(config.bert_hidden_dim, config.relation_vocab.size())

    def forward(self, inputs):
        encoder_output = self.encoder(inputs['bert_token'], inputs['bert_attention_mask'])[1]
        logits = self.linear(encoder_output)
        pred = torch.argmax(logits, dim=-1)
        return logits, pred
