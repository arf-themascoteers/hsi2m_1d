import torch
import torch.nn as nn

class Machine(nn.Module):
    def __init__(self):
        super(Machine, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv1d(1, 8, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(8, 16, 4),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(448, 1)
        )

    def forward(self, x):
        x = self.fc(x)
        return x
