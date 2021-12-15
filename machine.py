import torch
import torch.nn as nn

class Machine(nn.Module):
    def __init__(self):
        super(Machine, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv1d(1, 8, 5),
            nn.Flatten(),
            nn.Linear(968, 1)
        )

    def forward(self, x):
        x = self.fc(x)
        return x
