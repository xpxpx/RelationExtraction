import torch
import random
import jsonlines as jl


class BERTDataLoader:
    def __init__(self, config, file, batch_size, shuffle=False):
        self.file = file
        self.relation_vocab = config.relation_vocab
        self.bert_tokenizer = config.bert_tokenizer

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data = []
        self.index = 0

        with jl.open(file, 'r') as f:
            for line in f:
                self.data.append(self.build(line))

        if self.shuffle:
            random.shuffle(self.data)

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.data) // self.batch_size

    def __next__(self):
        if self.index >= len(self.data):
            if self.shuffle:
                random.shuffle(self.data)
            self.index = 0
            raise StopIteration
        else:
            current_data = self.data[self.index:self.index + self.batch_size]
            self.index += self.batch_size

            # padding
            max_bert_length = max([one['bert_length'] for one in current_data])
            bert_token = []
            bert_attention_mask = []
            relation = []

            for one in current_data:
                bert_token.append(one['bert_token'] + [self.bert_tokenizer.pad_token_id] * (max_bert_length - one['bert_length']))
                bert_attention_mask.append([1] * one['bert_length'] + [0] * (max_bert_length - one['bert_length']))
                relation.append(one['relation'])

            return {
                'bert_token': torch.tensor(bert_token),
                'bert_attention_mask': torch.tensor(bert_attention_mask)
            }, {
                'relation': torch.tensor(relation)
            }

    def build(self, line):
        token = line['token']
        relation = line['relation']

        bert_token = []
        for t in token:
            bert_token.extend(self.bert_tokenizer.tokenize(t))

        bert_token = [self.bert_tokenizer.cls_token] + bert_token + [self.bert_tokenizer.sep_token]

        return {
            'bert_token': self.bert_tokenizer.convert_tokens_to_ids(bert_token),
            'bert_length': len(bert_token),
            'relation': self.relation_vocab.get_index(relation)
        }
