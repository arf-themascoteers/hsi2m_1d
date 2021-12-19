import torch
import torch.nn as nn
from machine import Machine
import time
import numpy as np
import data_reader
import os
import train
import plotter
from data_reader import DataReader


def test():
    model = Machine()
    criterion = torch.nn.MSELoss(reduction='mean')
    #if not os.path.isfile("models/machine.h5"):
    #train.train()
    model = torch.load("models/machine.h5")

    dr = DataReader()
    _, _, x_test, y_test = dr.get_data()
    x_test = x_test.reshape(x_test.shape[0], -1)
    y_test = y_test.reshape(-1, 1)
    y_test_pred = model(x_test)
    loss = criterion(y_test_pred, y_test).item()
    print(f"Test Loss {loss:.2f}")

    #plotter.plot(y_test.detach().numpy(), y_test_pred.detach().numpy())


if __name__ == "__main__":
    test()
