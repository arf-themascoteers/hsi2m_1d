import torch
import torch.nn as nn


class JunkMachine(nn.Module):
    def __init__(self):
        super(JunkMachine, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(16,8),
            nn.LeakyReLU(),
            nn.Linear(8,1)
        )

    def forward(self, x):
        x = self.fc(x)
        return x
