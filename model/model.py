import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

class EcgAttention(nn.Module):
    def __init__(self, num_classes=12):
        super().__init__()
        self.in_channels = 12
        self.pool0 = nn.AvgPool1d(kernel_size=2)
        self.pool1 = nn.AvgPool1d(kernel_size=3)
        self.conv0 = nn.Conv1d(in_channels=self.in_channels,
                               out_channels=2*self.in_channels,
                               kernel_size=51,
                               groups=self.in_channels)
        self.bn0 = nn.BatchNorm1d(num_features=2*self.in_channels)

        self.conv1 = nn.Conv1d(in_channels=2*self.in_channels,
                               out_channels=4*self.in_channels,
                               kernel_size=26,
                               groups=self.in_channels)
        self.bn1 = nn.BatchNorm1d(num_features=4*self.in_channels)

        self.conv2 = nn.Conv1d(in_channels=4*self.in_channels,
                               out_channels=4*self.in_channels,
                               kernel_size=10,
                               groups=self.in_channels)
        self.bn2 = nn.BatchNorm1d(num_features=4*self.in_channels)

        self.conv3 = nn.Conv1d(in_channels=4*self.in_channels,
                               out_channels=5*self.in_channels,
                               kernel_size=8)
        self.bn3 = nn.BatchNorm1d(num_features=5*self.in_channels)

        self.conv4 = nn.Conv1d(in_channels=5*self.in_channels,
                               out_channels=5*self.in_channels,
                               kernel_size=4)
        self.bn4 = nn.BatchNorm1d(num_features=5*self.in_channels)

        self.conv5 = nn.Conv1d(in_channels=self.in_channels,
                               out_channels=self.in_channels // 2,
                               kernel_size=11)
        self.bn5 = nn.BatchNorm1d(num_features=self.in_channels // 2)
        self.pred = nn.Linear(in_features=6 * 730, out_features=num_classes)
        encoder_layer = nn.TransformerEncoderLayer(d_model=740, nhead=5, dim_feedforward=740*2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

    def forward(self, x):
        x = self.bn0(F.relu(self.conv0(x)))
        x = self.pool0(x)
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.pool0(x)
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.pool0(x)
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.pool0(x)
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.pool0(x)
        # Split channels
        x = x.view((x.shape[0], self.in_channels, -1)).transpose(0, 1)
        x = self.transformer_encoder(x).transpose(0, 1)
        x = self.bn5(F.relu(self.conv5(x)))
        x = x.view((x.shape[0], -1))
        x = self.pred(x)
        return x


if __name__ == '__main__':
    net = EcgAttention().double()
    x = torch.from_numpy(np.random.rand(3, 12, 5000)).double()
    y = net(x)
    print(y.shape)