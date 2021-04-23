import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from model import EcgAttention
from dataset import load_tenor_dataset


class TrainingPipeline:
    def __init__(self):
        self.device = torch.device(
                "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.epochs = 50
        self.model = EcgAttention(num_classes=6).to(self.device).double()
        self.loss = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
        self.train_ds, self.test_ds = load_tenor_dataset(self.device)
        print('Initialized successfully')

    def train(self):
        train_dl = DataLoader(self.train_ds, batch_size=128)
        for epoch in range(self.epochs):
            self.model.train()
            for x, y in train_dl:
                pred = self.model(x)
                loss = self.loss(pred, y)
                loss.backward()

                self.optimizer.step()

                self.optimizer.zero_grad()

            history = self.validate()
            self.scheduler.step(history['loss'])

    def train_step(self):
        pass

    def validate(self):
        self.model.eval()
        with torch.no_grad():
            guess = 0
            loss = 0
            n = 0
            i = 0
            for x, y in DataLoader(self.test_ds, batch_size=128):
                out = self.model(x)
                loss += self.loss(out, y)
                i += 1
                y_pred = (out > 0.5).int()
                diff = y - y_pred
                guess += (diff == 1).int().sum()
                n += y_pred.view(-1).size(0)

        res = {
            'acc': guess / n,
            'loss': loss / i
        }
        print(res)
        return res


if __name__ == '__main__':
    t = TrainingPipeline()
    t.train()
