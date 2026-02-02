import torch
import torch.nn as nn

class BFAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.att = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention = self.att(x)
        return x * attention
