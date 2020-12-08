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
        self.device = torch.device(f'cuda:{config.gpu}' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.dataloader = dataloader

        if config.optimizer == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=config.learning_rate)
        elif config.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        elif config.optimizer == 'AdamW':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        else:
            raise ValueError('Unknown optimizer.')

        self.crit = nn.CrossEntropyLoss()
        self.metrics = Metrics()

    def train(self, start_epoch=0):
        best_f1 = 0.0
        for epoch in range(start_epoch + 1, self.config.epochs + 1):
            self.model.train()
            pbar = tqdm(self.dataloader['train'], ncols=100)
            for step, (data, label) in enumerate(pbar):
                self.optimizer.zero_grad()
                # forward
                logits, pred = self.model(
                    filter_inputs(data, self.config.train_keys, self.device)
                )
                # backward
                loss = self.crit(logits, label['relation'].to(self.device))
                loss.backward()

                # gradient clip
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

                self.optimizer.step()

                pbar.set_description(f"[EPOCH]: {epoch}/{self.config.epochs} | [STEP]: {step}/{len(self.dataloader['train'])} | Loss: {loss.item():.4f}")

            # save
            state = {
                'net': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epoch': epoch
            }
            torch.save(state, Path(self.model_path, f'model-{epoch}.pkl'))

            # evaluate
            result = self.evaluate()
            print(f"Epoch: {epoch} | P: {result['precision']} | R: {result['recall']} | F1: {result['f1']}")
            if result['f1'] > best_f1:
                print('Get a new best result.')
                best_f1 = result['f1']
                # 0 mean best model
                torch.save(state, Path(self.model_path, 'model-0.pkl'))

            self.logger.write({
                'epoch': epoch,
                'precision': result['precision'],
                'recall': result['recall'],
                'f1': result['f1'],
                'best_f1': best_f1
            })

    def evaluate(self, dataloader_name='dev'):
        self.model.eval()
        with torch.no_grad():
            pbar = tqdm(self.dataloader[dataloader_name], ncols=100)
            for step, (data, label) in enumerate(pbar):
                pass
        pass
