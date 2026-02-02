import torch
import torch.nn as nn
from models.bfam import BFAM
from models.bnsm import BNSM

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = nn.Conv2d(3, 64, 3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, 3, padding=1)

        self.bfam = BFAM(128)
        self.bnsm = BNSM()

        self.dec1 = nn.Conv2d(128, 64, 3, padding=1)
        self.dec2 = nn.Conv2d(64, 1, 1)

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x):
        x1 = self.pool(torch.relu(self.enc1(x)))
        x2 = self.pool(torch.relu(self.enc2(x1)))

        x2 = self.bfam(x2)
        x2 = self.bnsm(x2)

        x = self.up(torch.relu(self.dec1(x2)))
        x = torch.sigmoid(self.dec2(x))
        return x
