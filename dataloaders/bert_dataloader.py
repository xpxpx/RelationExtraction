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
            bert_head_index = []
            bert_tail_index = []
            bert_head_start_position = []
            bert_tail_start_position = []
            relation = []

            for one in current_data:
                bert_token.append(one['bert_token'] + [self.bert_tokenizer.pad_token_id] * (max_bert_length - one['bert_length']))
                bert_attention_mask.append([1] * one['bert_length'] + [0] * (max_bert_length - one['bert_length']))
                bert_head_index.append(one['bert_head_index'] + [0] * (max_bert_length - one['bert_length']))
                bert_tail_index.append(one['bert_tail_index'] + [0] * (max_bert_length - one['bert_length']))
                bert_head_start_position.append(one['bert_head_start_position'])
                bert_tail_start_position.append(one['bert_tail_start_position'])
                relation.append(one['relation'])

            return {
                'bert_token': torch.tensor(bert_token),
                'bert_attention_mask': torch.tensor(bert_attention_mask),
                'bert_head_index': torch.tensor(bert_head_index),
                'bert_tail_index': torch.tensor(bert_tail_index),
                'bert_head_start_position': torch.tensor(bert_head_start_position),
                'bert_tail_start_position': torch.tensor(bert_tail_start_position)
            }, {
                'relation': torch.tensor(relation)
            }

    def build(self, line):
        bert_token = line['token']
        relation = line['relation']
        head_start, head_end = line['head'][1][0], line['head'][1][-1]
        tail_start, tail_end = line['tail'][1][0], line['tail'][1][-1]

        bert_token = [self.bert_tokenizer.cls_token] + bert_token + [self.bert_tokenizer.sep_token]
        head_index = [0] * len(bert_token)
        tail_index = [0] * len(bert_token)

        # +1 means [CLS]
        head_index[head_start + 1:head_end + 2] = [1] * (head_end - head_start + 1)
        tail_index[tail_start + 1:tail_end + 2] = [1] * (tail_end - tail_start + 1)

        return {
            'bert_token': self.bert_tokenizer.convert_tokens_to_ids(bert_token),
            'bert_length': len(bert_token),
            'bert_head_index': head_index,
            'bert_tail_index': tail_index,
            'bert_head_start_position': [head_start + 1],
            'bert_tail_start_position': [tail_start + 1],
            'bert_head_end_position': [head_end + 1],
            'bert_tail_end_position': [tail_end + 1],
            'relation': self.relation_vocab.get_index(relation)
        }
