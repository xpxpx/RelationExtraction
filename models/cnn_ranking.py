import math
import torch
import torch.nn as nn
from layers.embedding import Embedding
from layers.cnn_encoder import CNNEncoder
from utils.constant import PAD, MAX_LEN


class CNNRanking(nn.Module):
    def __init__(self, config):
        super(CNNRanking, self).__init__()
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

        self.linear = nn.Linear(config.hidden_dim, config.output_dim)

        self.relation_embedding = nn.Parameter(torch.Tensor(config.output_dim, config.relation_vocab.size()))
        scale = math.sqrt(6 / (config.relation_vocab.size() + config.output_dim))
        nn.init.uniform_(self.relation_embedding.data, -scale, scale)

        self.input_drop = nn.Dropout(config.input_dropout)
        self.hidden_drop = nn.Dropout(config.hidden_dropout)

        self.gamma = config.gamma
        self.m_plus = config.m_plus
        self.m_minus = config.m_minus

    def forward(self, inputs, label):
        word_embed = self.word_emb(inputs['word'])
        head_position_embed = self.head_position_emb(inputs['head_position'])
        tail_position_embed = self.tail_position_emb(inputs['tail_position'])
        total_embed = self.input_drop(torch.cat([word_embed, head_position_embed, tail_position_embed], dim=-1))

        hidden = self.encoder(total_embed)
        hidden = self.hidden_drop(hidden)
        hidden = torch.tanh(self.linear(hidden))

        positive_score = torch.sum(hidden * torch.index_select(self.relation_embedding, 1, label['relation']).t(), dim=-1)
        total_negative_score = torch.matmul(hidden, self.relation_embedding)

        top2_negative_score, top2_negative_index = torch.topk(total_negative_score, 2)
        first_negative_score, second_negative_score = top2_negative_score[:, 0], top2_negative_score[:, 1]
        first_negative_index, second_negative_index = top2_negative_index[:, 0], top2_negative_index[:, 1]

        negative_score = torch.where(first_negative_index.eq(label['relation']), second_negative_score, first_negative_score)
        loss = torch.log(1 + torch.exp(self.gamma * (self.m_plus - positive_score))) + torch.log(1 + torch.exp(self.gamma * (self.m_minus + negative_score)))
        return loss.mean()

    def predict(self, inputs):
        word_embed = self.word_emb(inputs['word'])
        head_position_embed = self.head_position_emb(inputs['head_position'])
        tail_position_embed = self.tail_position_emb(inputs['tail_position'])
        total_embed = self.input_drop(torch.cat([word_embed, head_position_embed, tail_position_embed], dim=-1))

        hidden = self.encoder(total_embed)
        hidden = self.hidden_drop(hidden)
        hidden = torch.tanh(self.linear(hidden))

        score = torch.matmul(hidden, self.relation_embedding)
        pred = torch.argmax(score, dim=-1)
        return pred
