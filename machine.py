import torch
import torch.nn as nn

class Machine(nn.Module):
    def __init__(self):
        super(Machine, self).__init__()
        self.fc1 = nn.Linear(125,16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.relu(x1)
        x3 = self.fc2(x2)
        return x3
