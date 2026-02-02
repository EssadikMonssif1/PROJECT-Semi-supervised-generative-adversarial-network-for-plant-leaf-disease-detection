import torch
import torch.nn as nn

class BNSM(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

    def forward(self, x):
        smooth = self.pool(x)
        return x - smooth
