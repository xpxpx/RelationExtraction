import json
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
            self.build_word_vocab()
            self.build_relation_vocab(self.relation_file)
            self.load_pretrain_embedding(self.embedding_file)
            self.save(vocab_file)
        else:
            self.load(vocab_file)

    def build_word_vocab(self):
        pass

    def build_relation_vocab(self):
        pass

    def load_pretrain_embedding(self):
        pass

    def save(self, vocab_file):
        pass

    def load(self, vocab_file):
        pass
