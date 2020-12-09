import torch
import torch.nn as nn
from layers.embedding import Embedding
from layers.cnn_encoder import CNNEncoder
from utils.constant import PAD, MAX_LEN


class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        self.word_emb = Embedding(
            config.word_vocab.size(),
            config.word_embedding_dim,
            padding_idx=config.word_vocab.get_index(PAD),
            pretrain_embedding=config.pretrain_word_embedding
        )
        self.head_position_emb = Embedding(
            config.head_position_vocab.size(),
            config.position_embedding_dim,
            padding_idx=config.head_position_vocab.get_index(MAX_LEN)
        )
        self.tail_position_emb = Embedding(
            config.tail_position_vocab.size(),
            config.position_embedding_dim,
            padding_idx=config.tail_position_vocab.get_index(MAX_LEN)
        )

        self.encoder = CNNEncoder(
            config.word_embedding_dim + 2 * config.position_embedding_dim,
            config.hidden_dim,
            kernel_size=config.kernel_size
        )

        self.linear = nn.Linear(config.hidden_dim, config.relation_vocab.size())

        self.input_drop = nn.Dropout(config.input_dropout)
        self.hidden_drop = nn.Dropout(config.hidden_dropout)

    def forward(self, inputs):
        word_embed = self.word_emb(inputs['word'])
        head_position_embed = self.head_position_emb(inputs['head_position'])
        tail_position_embed = self.tail_position_emb(inputs['tail_position'])
        total_embed = self.input_drop(torch.cat([word_embed, head_position_embed, tail_position_embed], dim=-1))

        hidden = self.encoder(total_embed)
        hidden = self.hidden_drop(hidden)

        logits = self.linear(hidden)
        pred = torch.argmax(logits, dim=-1)
        return logits, pred
