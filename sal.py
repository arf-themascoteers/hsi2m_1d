import torch
import matplotlib.pyplot as plt
from data_reader import DataReader

def test(model, data, y_true):
    model = torch.load("models/machine.h5")
    model.train()
    data = data.clone()
    data.requires_grad = True
    y_pred = model(data)
    loss = criterion(y_pred, y_true)
    accuracy = 1/loss
    accuracy.backward()
    x = model.la
    std = torch.std(x)
    mean = torch.mean(x)

    x = (x-mean)/std
    x = x - torch.min(x)
    x = x.squeeze().squeeze()
    plt.plot(x)
    plt.show()

if __name__ == "__main__":
    model = torch.load("models/machine.h5")
    criterion = torch.nn.MSELoss(reduction='mean')
    model.eval()
    dr = DataReader()
    _, _, x_test, y_test = dr.get_data()
    y_test = y_test.reshape(-1, 1)
    items = 0
    for i in range(len(y_test)):
        current_x = x_test[i].unsqueeze(dim=0)
        current_y = y_test[i].unsqueeze(dim=0)
        y_test_pred = model(current_x)
        loss = criterion(y_test_pred, current_y).item()
        if loss < 0.1:
            test(model, current_x, current_y)
            items += 1
            if items >= 5:
                break


