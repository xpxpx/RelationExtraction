from collections import Counter
from utils.constant import UNK


class Vocab:
    def __init__(self, name, fixed_instance=[]):
        self.name = name
        self.keep_growing = True
        self.instance2index = {}
        self.index2instance = {}
        self.instance2count = Counter()
        self.fixed_instance = fixed_instance
        self.next_index = 0

        self.add_fixed_instance()

    def add_fixed_instance(self):
        for instance in self.fixed_instance:
            self.instance2index[instance] = self.next_index
            self.index2instance[self.next_index] = instance
            self.next_index += 1

    def add(self, instance):
        self.instance2count.update([instance])
        if instance not in self.instance2index:
            self.instance2index[instance] = self.next_index
            self.index2instance[self.next_index] = instance
            self.next_index += 1

    def get_index(self, instance):
        if instance not in self.instance2index:
            return self.instance2index[UNK]
        else:
            return self.instance2index[instance]

    def get_instance(self, index):
        if index in self.index2instance:
            return self.index2instance[index]
        else:
            raise ValueError(f"wrong index for {self.name} vocab, can't get instance.")

    def size(self):
        return self.next_index

    def shrink_by_count(self, min_count):
        self.instance2index = {}
        self.index2instance = {}
        self.next_index = 0

        self.add_fixed_instance()

        for instance, count in self.instance2count.items():
            if count >= min_count:
                self.instance2index[instance] = self.next_index
                self.index2instance[self.next_index] = instance
                self.next_index += 1

    def shrink_by_size(self, size):
        self.instance2index = {}
        self.index2instance = {}
        self.next_index = 0

        self.add_fixed_instance()

        instance2count = sorted([(instance, count) for instance, count in self.instance2count.items()], key=lambda x: x[1], reverse=True)
        for instance, count in instance2count[:size]:
            self.instance2index[instance] = self.next_index
            self.index2instance[self.next_index] = instance
            self.next_index += 1


class PositionVocab(Vocab):
    def __init__(self, name, fixed_instance=[]):
        super(PositionVocab, self).__init__(name, fixed_instance=fixed_instance)

    def get_index(self, instance):
        if instance not in self.instance2index:
            return self.find_closed_index(instance)
        else:
            return self.instance2index[instance]

    def find_closed_index(self, instance):
        result = sorted([(abs(instance - key), index) for key, index in self.instance2index.items()], key=lambda x: x[0])
        return result[0][1]
