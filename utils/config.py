import json
import pickle
import numpy as np
import jsonlines as jl
from utils.vocab import Vocab, PositionVocab
from utils.constant import PAD, UNK, MAX_LEN


class Config:
    def __init__(self, config_file):
        self.load_config_file(config_file)
        self.raw_config_data = json.load(open(config_file, 'r', encoding='utf-8'))

    def load_config_file(self, config_file):
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        for k, v in config.items():
            setattr(self, k, v)

    def build_vocab(self, vocab_file, build_vocab):
        self.word_vocab = Vocab('word', [PAD, UNK])
        self.relation_vocab = Vocab('relation')
        self.head_position_vocab = PositionVocab('head_position', [MAX_LEN])
        self.tail_position_vocab = PositionVocab('tail_position', [MAX_LEN])
        self.pretrain_word_embedding = None

        if build_vocab is True:
            self.build_word_vocab(self.embedding_file)
            self.build_relation_vocab(self.relation_file)
            self.build_position_vocab(self.train_file)
            self.load_pretrain_word_embedding(self.embedding_file)
            self.save(vocab_file)
        else:
            self.load(vocab_file)

    def build_word_vocab(self, file):
        with jl.open(file, 'r') as f:
            for line in f:
                self.word_vocab.add(line['token'])

    def build_relation_vocab(self, file):
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                self.relation_vocab.add(line.strip())

    def build_position_vocab(self, file):
        with jl.open(file, 'r') as f:
            for line in f:
                head_start, head_end = line['head'][1][0], line['head'][1][-1]
                tail_start, tail_end = line['tail'][1][0], line['tail'][1][-1]

                for index in range(len(line['token'])):
                    if index < head_start:
                        self.head_position_vocab.add(index - head_start)
                    elif index > head_end:
                        self.head_position_vocab.add(index - head_end)
                    else:
                        self.head_position_vocab.add(0)

                    if index < tail_start:
                        self.tail_position_vocab.add(index - tail_start)
                    elif index > tail_end:
                        self.tail_position_vocab.add(index - tail_end)
                    else:
                        self.tail_position_vocab.add(0)

    def load_pretrain_word_embedding(self, embedding_file):
        scale = np.sqrt(3.0 / self.word_embedding_dim)
        pretrain_embedding = np.random.uniform(-scale, scale, [self.word_vocab.size(), self.word_embedding_dim])
        with jl.open(embedding_file, 'r') as f:
            for line in f:
                if line['token'] in self.word_vocab.instance2index:
                    pretrain_embedding[self.word_vocab.get_index(line['token'])] = line['vec']

        self.pretrain_word_embedding = pretrain_embedding

    def save(self, vocab_file):
        pickle.dump({
            'word_vocab': self.word_vocab,
            'relation_vocab': self.relation_vocab,
            'head_position_vocab': self.head_position_vocab,
            'tail_position_vocab': self.tail_position_vocab,
            'pretrain_word_embedding': self.pretrain_word_embedding
        }, open(vocab_file, 'wb'))

    def load(self, vocab_file):
        vocab_data = pickle.load(open(vocab_file, 'rb'))
        self.word_vocab = vocab_data['word_vocab']
        self.relation_vocab = vocab_data['relation_vocab']
        self.head_position_vocab = vocab_data['head_position_vocab']
        self.tail_position_vocab = vocab_data['tail_position_vocab']
        self.pretrain_word_embedding = vocab_data['pretrain_word_embedding']
