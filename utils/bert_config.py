import pickle
from transformers import AutoTokenizer
from utils.vocab import Vocab
from .config import Config


class BERTConfig(Config):
    def __init__(self, config_file):
        super(BERTConfig, self).__init__(config_file)

    def build_vocab(self, vocab_file, build_vocab):
        self.relation_vocab = Vocab('relation')
        self.bert_tokenizer = None

        if build_vocab is True:
            self.build_relation_vocab(self.relation_file)
            self.build_tokenizer(self.bert_file)
            self.save(vocab_file)
        else:
            self.load(vocab_file)

    def build_tokenizer(self, bert_file):
        self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_file)

    def save(self, vocab_file):
        pickle.dump({
            'relation_vocab': self.relation_vocab,
            'bert_tokenizer': self.bert_tokenizer
        }, open(vocab_file, 'wb'))

    def load(self, vocab_file):
        vocab_data = pickle.load(open(vocab_file, 'rb'))
        self.relation_vocab = vocab_data['relation_vocab']
        self.bert_tokenizer = vocab_data['bert_tokenizer']
