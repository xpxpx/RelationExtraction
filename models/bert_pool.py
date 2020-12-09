import torch
import torch.nn as nn
from layers.bert_encoder import BERTEncoder


class BERTPool(nn.Module):
    def __init__(self, config):
        super(BERTPool, self).__init__()
        self.encoder = BERTEncoder(config.bert_file)
        self.linear = nn.Linear(config.bert_hidden_dim * 2, config.relation_vocab.size())

    def forward(self, inputs):
        encoder_output = self.encoder(inputs['bert_token'], inputs['bert_attention_mask'])[0]
        head_output = self.pooling(encoder_output, inputs['bert_head_index'])
        tail_output = self.pooling(encoder_output, inputs['bert_tail_index'])
        total_output = torch.cat([head_output, tail_output], dim=-1)
        logits = self.linear(total_output)
        pred = torch.argmax(logits, dim=-1)
        return logits, pred

    @staticmethod
    def pooling(inputs, index, pooling='max'):
        if pooling == 'max':
            output = inputs.masked_fill(index.eq(0).unsqueeze(2), float('-inf'))
            output, _ = torch.max(output, dim=1)
        elif pooling == 'avg':
            output = inputs.masked_fill(index.eq(0).unsqueeze(2), 0.0)
            output = torch.sum(output, dim=1) / torch.sum(index, dim=1).unsqueeze(1)
        else:
            raise ValueError('Unknown pooling.')
        return output
