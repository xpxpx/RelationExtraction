import torch
import random
import jsonlines as jl
import unicodedata
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def filter_inputs(inputs, key, device=torch.device('cpu')):
    outputs = dict()
    for k, v in inputs.items():
        if k in key:
            outputs[k] = v.to(device)
        else:
            outputs[k] = v
    return outputs


def word_norm(word):
    word = unicodedata.normalize('NFKD', word)
    return "".join(c for c in word if not unicodedata.combining(c))


def plot_CDF(values):
    counter = Counter(values)
    sorted_value_count = sorted([(k, v) for k, v in counter.items()], key=lambda x: x[0])
    sorted_value = [k for k, v in sorted_value_count]
    sorted_count = [v for k, v in sorted_value_count]
    total_count = sum(sorted_count)

    add_count = [sorted_count[0]]
    for v in sorted_count[1:]:
        add_count.append(v + add_count[-1])

    norm_add_count = [v / total_count for v in add_count]
    plt.plot(sorted_value, norm_add_count)
    plt.grid()
    plt.show()


# only for non-bert model
def compute_embedding_coverage(input_file, embedding_file, uncased=True):
    token = set()
    with jl.open(input_file, 'r') as f:
        for line in f:
            if uncased is True:
                token.update([t.lower() for t in line['token']])
            else:
                token.update(line['token'])

    vocab = set()
    with jl.open(embedding_file, 'r') as f:
        for line in f:
            vocab.add(line['token'])

    print(f'Token Coverage: {len(token & vocab) / len(token):.4f}')
