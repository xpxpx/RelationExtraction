import json
from utils.vocab import Vocab
from utils.constant import PAD, UNK


class Config:
    def __init__(self, config_file):
        self.load_config_file(config_file)
        self.raw_config_data = json.load(open(config_file, 'r', encoding='utf-8'))

    def load_config_file(self, config_file):
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        for k, v in config.items():
            setattr(self, k, v)

    def build_vocab(self, vocab_file, build_vocab):
        pass

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
