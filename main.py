import argparse
import utils
import dataloaders
import models
import trainers
from utils.function import set_random_seed


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
    # set random seed
    set_random_seed(config.seed)

    # build dataloader
    train_dataloader = getattr(dataloaders, config.dataloader)(config, config.train_file, config.batch_size, shuffle=True)
    dev_dataloader = getattr(dataloaders, config.dataloader)(config, config.dev_file, 50, shuffle=False)
    test_dataloader = getattr(dataloaders, config.dataloader)(config, config.test_file, 50, shuffle=False)

    # build model
    model = getattr(models, config.model)(config)

    # build trainer
    trainer = getattr(trainers, config.trainer)(config, model, dataloader={'train': train_dataloader, 'dev': dev_dataloader, 'test': test_dataloader})

    trainer.train()


def evaluate(config):
    pass


def main():
    args = vars(parse_args())
    if args['run'] == 'train':
        config = getattr(utils, args['config'])(args['config_file'])
        # update config by input
        config.gpu = args['gpu']
        config.seed = args['seed']
        config.alias += '_' + str(args['seed'])
        config.build_vocab(args['vocab_file'], args['build_vocab'])
        train(config)
    elif args['run'] == 'evaluate':
        pass
    else:
        raise ValueError('Unknown run operate.')


if __name__ == '__main__':
    main()
