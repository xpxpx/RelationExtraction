import torch
import random
import jsonlines as jl
from utils.function import word_norm
from utils.constant import PAD, MAX_LEN


class DataLoader:
    def __init__(self, config, file, batch_size, shuffle=False):
        self.file = file
        self.word_vocab = config.word_vocab
        self.relation_vocab = config.relation_vocab
        self.head_position_vocab = config.head_position_vocab
        self.tail_position_vocab = config.tail_position_vocab
        self.word_uncased = config.word_uncased
        self.word_norm = config.word_norm

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
            max_length = max([one['length'] for one in current_data])
            token = []
            length = []
            head_position = []
            tail_position = []
            relation = []

            for one in current_data:
                token.append(one['token'] + [self.word_vocab.get_index(PAD)] * (max_length - one['length']))
                length.append(one['length'])
                head_position.append(one['head_position'] + [self.head_position_vocab.get_index(MAX_LEN)] * (max_length - one['length']))
                tail_position.append(one['tail_position'] + [self.tail_position_vocab.get_index(MAX_LEN)] * (max_length - one['length']))
                relation.append(one['relation'])

            return {
                'token': torch.tensor(token),
                'length': torch.tensor(length),
                'head_position': torch.tensor(head_position),
                'tail_position': torch.tensor(tail_position)
            }, {
                'relation': torch.tensor(relation)
            }

    def build(self, line):
        token = line['token']
        relation = line['relation']
        head_start, head_end = line['head'][1][0], line['head'][1][-1]
        tail_start, tail_end = line['tail'][1][0], line['tail'][1][-1]

        if self.word_norm:
            token = [word_norm(t) for t in token]

        if self.word_uncased:
            token = [t.lower() for t in token]

        head_position = []
        tail_position = []
        for index in range(len(token)):
            if index < head_start:
                head_position.append(index - head_start)
            elif index > head_end:
                head_position.append(index - head_end)
            else:
                head_position.append(0)

            if index < tail_start:
                tail_position.append(index - tail_start)
            elif index > tail_end:
                tail_position.append(index - tail_end)
            else:
                tail_position.append(0)

        return {
            'token': [self.word_vocab.get_index(t) for t in token],
            'length': len(token),
            'head_position': [self.head_position_vocab.get_index(p) for p in head_position],
            'tail_position': [self.tail_position_vocab.get_index(p) for p in tail_position],
            'relation': self.relation_vocab.get_index(relation)
        }
