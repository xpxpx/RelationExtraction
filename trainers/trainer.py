import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
from utils.metrics import Metrics
from utils.function import filter_inputs
from .base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(self, config, model, dataloader=None):
        super(Trainer, self).__init__(config)

    def train(self, start_epoch=0):
        pass

    def evaluate(self, dataloader_name='dev'):
        pass
