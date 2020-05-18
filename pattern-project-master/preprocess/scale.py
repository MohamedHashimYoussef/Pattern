from sklearn.preprocessing import MinMaxScaler
import numpy as np


def scale(arr, test):
    scaler = MinMaxScaler()
    All = [*arr, *test]
    scaler.fit(All)
    print(" => Data was scaled to (0, 1)")
    return scaler.transform(arr), scaler.transform(test)
