import json
import pickle
import jsonlines
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


class BaseTrainer:
    def __init__(self, config):
        self.config = config
        # build checkpoint path
        checkpoint_path = Path('checkpoint', self.config.alias)
        model_path = Path(checkpoint_path, 'model')
        result_path = Path(checkpoint_path, 'result')

        checkpoint_path.mkdir(parents=True, exist_ok=True)
        model_path.mkdir(parents=True, exist_ok=True)
        result_path.mkdir(parents=True, exist_ok=True)

        # save config file for further eval and retrain
        pickle.dump(self.config, open(Path(checkpoint_path, 'config.pkl'), 'wb'))
        json.dump(self.config.raw_config_data, open(Path(checkpoint_path, 'raw_config.json'), 'w'), indent=4)

        self.checkpoint_path = checkpoint_path
        self.model_path = model_path
        self.result_path = result_path

        self.logger = jsonlines.Writer(open(Path(checkpoint_path, 'log.jl'), 'a'), flush=True)
        self.writer = SummaryWriter(log_dir='./runs/' + self.config.alias)
