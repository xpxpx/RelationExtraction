from transformers import AutoTokenizer
from .config import Config


class BERTConfig(Config):
    def __init__(self, config_file):
        super(BERTConfig, self).__init__(config_file)

    def build_vocab(self, vocab_file, build_vocab):
        pass

    def build_tokenizer(self):
        pass

    def save(self, vocab_file):
        pass

    def load(self, vocab_file):
        pass
