import torch
from concise_machine import ConciseMachine
from data_reader import DataReader


def train():
    model = ConciseMachine()
    criterion = torch.nn.MSELoss(reduction='mean')
    optimiser = torch.optim.Adam(model.parameters(), lr=0.1)
    dr = DataReader()
    x_train, y_train, _, _ = dr.get_concise_data()
    x_train = x_train.reshape(x_train.shape[0], -1)
    y_train = y_train.reshape(-1,1)
    for t in range(10):
        y_train_pred = model(x_train)
        loss = criterion(y_train_pred, y_train)
        print("Epoch ", t, "MSE: ", loss.item())
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    torch.save(model, "models/concise_machine.h5")


def test():
    model = ConciseMachine()
    criterion = torch.nn.MSELoss(reduction='mean')
    #if not os.path.isfile("models/machine.h5"):
    #train.train()
    model = torch.load("models/concise_machine.h5")

    dr = DataReader()
    _, _, x_test, y_test = dr.get_concise_data()
    x_test = x_test.reshape(x_test.shape[0], -1)
    y_test = y_test.reshape(-1, 1)
    y_test_pred = model(x_test)
    loss = criterion(y_test_pred, y_test).item()
    print(f"Test Loss {loss:.5f}")

    #plotter.plot(y_test.detach().numpy(), y_test_pred.detach().numpy())


if __name__ == "__main__":
    train()
    test()
