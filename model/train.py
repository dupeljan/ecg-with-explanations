import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from model import EcgAttention
from dataset import load_tenor_dataset
from utils import AverageMeter


CHECKPOINT_SAVE_PATH = '../checkpoints/best_EcgAttention_v0.1.pth'
CHECKPOINT_LOAD_PATH = '../checkpoints/best_EcgAttention.pth'


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
    def __init__(self, checkpoint: str = None, resume_epoch: int = 0):
        self.device = torch.device(
                "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.epochs = 250
        self.begin_epoch = 0
        self.batch_size = 256
        self.best_acc = 0.
        self.model = EcgAttention(num_classes=6).to(self.device).double()
        if checkpoint:
            print(f'Loading from checkpoint {checkpoint}')
            self.model.load_state_dict(torch.load(checkpoint))
            self.begin_epoch = resume_epoch
            print('Loaded successfully!')

        self.loss = FocalLoss(self.device, alpha=10, gamma=5)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-6, weight_decay=0.1)
        #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-5, momentum=0.9, weight_decay=0.1)
        self.scheduler_exp = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, 0.999)
        self.scheduler_plateu = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
        print('Loading dataset...')
        self.train_ds, self.test_ds = load_tenor_dataset(self.device)

        self.mean_train_loss = AverageMeter()
        self.mean_test_loss = AverageMeter()
        self.mean_acc = AverageMeter()
        print('Initialized successfully')
    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def train(self):
        tenorboard = SummaryWriter()
        train_dl = DataLoader(self.train_ds, batch_size=self.batch_size)
        for epoch in range(self.begin_epoch, self.epochs):
            print(f'epoch {epoch}')
            self.model.train()
            for x, y in tqdm(train_dl):
                pred = self.model(x)
                loss = self.loss(pred, y)
                print(f'loss: {loss.item()}, lr {self.get_lr()}')
                self.mean_train_loss.update(loss.item())
                loss.backward()

                self.optimizer.step()

                self.optimizer.zero_grad()
                #self.scheduler.step()#history['loss'])
            history = self.validate()
            print(f'Mean train loss {self.mean_train_loss.avg}')
            print(history)

            # Save checkpoint
            if history['acc'] > self.best_acc:
                print('Save best result!')
                self.best_acc = history['acc']
                torch.save(self.model.state_dict(), CHECKPOINT_SAVE_PATH)

            # Write to tensorboard
            tenorboard.add_scalar('Loss/train', self.mean_train_loss.avg, epoch)
            tenorboard.add_scalar('Loss/test', history['loss'], epoch)
            tenorboard.add_scalar('Acc/test', history['acc'], epoch)
            tenorboard.add_scalar('LR', self.get_lr(), epoch)
            self.scheduler_exp.step()
            #self.scheduler_plateu.step(history['loss'])

            self.mean_train_loss.reset()

    def train_step(self):
        pass

    def validate(self):
        self.model.eval()
        with torch.no_grad():
            guess = 0
            loss = AverageMeter()
            n = 0
            i = 0
            for x, y in tqdm(DataLoader(self.test_ds, batch_size=self.batch_size)):
                out = self.model(x)
                loss.update(self.loss(out, y))
                i += 1
                y_pred = (out > 0.).int()
                guess += sum([int(all(y[i] == y_pred[i])) for i in range(len(y))])
                n += y_pred.size(0)

        res = {
            'acc': guess / n,
            'loss': loss.avg
        }
        return res


if __name__ == '__main__':
    t = TrainingPipeline(CHECKPOINT_LOAD_PATH, 143)
    t.train()

