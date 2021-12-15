import torch
import torch.nn as nn

class Machine(nn.Module):
    def __init__(self):
        super(Machine, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv1d(1, 8, 5, stride=2),
            nn.LeakyReLU(),
            nn.MaxPool1d(3),
            # nn.Conv1d(8, 16, 5),
            # nn.LeakyReLU(),
            # nn.MaxPool1d(3),
            nn.Flatten(),
            nn.Linear(136, 1)
        )

    def forward(self, x):
        return self.fc(x)
