import torch
import torch.nn as nn
from layers.bert_encoder import BERTEncoder


class BERTState(nn.Module):
    def __init__(self, config):
        super(BERTState, self).__init__()
        self.encoder = BERTEncoder(config.bert_file)
        self.linear = nn.Linear(config.bert_hidden_dim * 2, config.relation_vocab.size())

    def forward(self, inputs):
        encoder_output = self.encoder(inputs['bert_token'], inputs['bert_attention_mask'])[0]
        head_start_output = self.gather(encoder_output, inputs['bert_head_start_position'])
        tail_start_output = self.gather(encoder_output, inputs['bert_tail_start_position'])
        total_output = torch.cat([head_start_output, tail_start_output], dim=-1)
        logits = self.linear(total_output)
        pred = torch.argmax(logits, dim=-1)
        return logits, pred

    @staticmethod
    def gather(inputs, index):
        batch_size = inputs.size(0)
        hidden_dim = inputs.size(1)
        output = torch.gather(inputs, 1, index.unsqueeze(2).expand(batch_size, 1, hidden_dim))
        return output.squeeze(1)
