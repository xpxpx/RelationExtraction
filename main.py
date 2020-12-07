import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Deep Learning for Relation Extraction')
    parser.add_argument('--config', type=str, default='Config', help='')
    parser.add_argument('--config_file', type=str, help='')
    parser.add_argument('--vocab_file', type=str, help='')
    parser.add_argument('--build_vocab', action='store_true', help='')
    parser.add_argument('--run', type=str, choices=['train', 'evaluate'], help='')
    parser.add_argument('--seed', type=int, default=123, help='')
    parser.add_argument('--gpu', type=int, default=0, help='')
    return parser.parse_args()


def train(config):
    pass


def evaluate(config):
    pass


def main():
    pass

if __name__ == '__main__':
    main()
