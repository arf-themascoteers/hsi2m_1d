import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from scipy import io
import sklearn.model_selection

class DataReader:
    def __init__(self):
        self.NUM_BANDS = 125
        self.raw_data = pd.read_csv("data/moisture.csv")
        self.moisture = torch.zeros(len(self.raw_data), dtype=torch.float32)
        self.bands = torch.zeros(len(self.raw_data), 1, self.NUM_BANDS)
        for index, row in self.raw_data.iterrows():
            self.moisture[index] = row[2]
            self.bands[index, 0] = torch.tensor(row[4:])

        self.moisture_scaler = MinMaxScaler(feature_range=(0, 1))

        temporary_moisture = self.moisture.reshape(-1,1)
        temporary_moisture = self.moisture_scaler.fit_transform(temporary_moisture)
        self.moisture = torch.tensor(temporary_moisture[:,0], dtype=torch.float32)
        self.data_size = len(self.moisture)

        self.x_train, self.x_test, self.y_train, self.y_test = sklearn.model_selection.train_test_split(self.bands, self.moisture, test_size=0.2,random_state=11)

    def get_data(self):
        return self.x_train, self.y_train, self.x_test, self.y_test

    def get_actual_bands(self):
        return self.bands[:,0,:]

if __name__ == "__main__":
    dr = DataReader()
    x_train, y_train, x_test, y_test = dr.get_data()
    print(len(x_train))
    print(len(y_train))
    print(len(x_test))
    print(len(y_test))
    print(len(dr.moisture))
    print(len(dr.bands))
    print(x_train[0])
    print(y_train[0])

