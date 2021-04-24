import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from model import EcgAttention
from dataset import load_tenor_dataset


class FocalLoss(nn.Module):

    def __init__(self, device, alpha=1.0, gamma=1.0):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha, dtype=torch.float32).to(device)
        self.gamma = torch.tensor(gamma, dtype=torch.float32).to(device)
        self.eps = 1e-6
        self.bcewithlogits = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, y_pred, y_true):
        probs = torch.sigmoid(y_pred)
        log_probs = self.bcewithlogits(y_pred, y_true)

        focal_loss = self.alpha * (y_true * ((1 - probs).pow(self.gamma)) +
                                    (1 - y_true) * probs.pow(self.gamma)) * log_probs
        focal_loss = torch.sum(focal_loss, dim=1)
        return focal_loss.mean()

class TrainingPipeline:
    def __init__(self):
        self.device = torch.device(
                "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.epochs = 50
        self.batch_size = 256
        self.model = EcgAttention(num_classes=6).to(self.device).double()
        self.loss = FocalLoss(self.device, alpha=10, gamma=5)#nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5, weight_decay=0.1)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, 0.99)
        #self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
        self.train_ds, self.test_ds = load_tenor_dataset(self.device)
        print('Initialized successfully')

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def train(self):
        train_dl = DataLoader(self.train_ds, batch_size=self.batch_size)
        for epoch in range(self.epochs):
            print(f'epoch {epoch}')
            self.model.train()
            for x, y in tqdm(train_dl):
                pred = self.model(x)
                loss = self.loss(pred, y)
                print(f'loss: {loss.item()}, lr {self.get_lr()}')
                loss.backward()

                self.optimizer.step()

                self.optimizer.zero_grad()

            history = self.validate()
            self.scheduler.step()#history['loss'])

    def train_step(self):
        pass

    def validate(self):
        self.model.eval()
        with torch.no_grad():
            guess = 0
            loss = 0
            n = 0
            i = 0
            for x, y in DataLoader(self.test_ds, batch_size=self.batch_size):
                out = self.model(x)
                loss += self.loss(out, y)
                i += 1
                y_pred = (out > 0.).int()
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
