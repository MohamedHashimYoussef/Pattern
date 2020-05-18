from preprocess.removeOutliers import remove_outliers
from preprocess.scale import scale
from preprocess.pca import pca
import numpy as np



def pre_process(arr, test, y=[]):
    test = test.to_numpy()
    arr  = arr.to_numpy()
    arr, y = remove_outliers(arr, y)
    #arr, test = pca(arr, test)
    arr, test = scale(arr, test)

    return arr, test, y

