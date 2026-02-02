import torch 
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),  # réduit à (batch, 128, 1, 1)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, img, mask):
        x = torch.cat([img, mask], dim=1)
        x = self.conv(x)
        x = self.fc(x)
        return x
