from sklearn.decomposition import PCA
import numpy as np
from configs import *


def pca(arr, test):
    reducer = reducerPCA
    arr = np.array(arr)
    p = PCA(n_components=reducer, random_state=randomstate)
    p.fit(arr)
    print(f" => [PCA applied] Reduced the number of components to ( {reducer} ) with score: { p.singular_values_}", end="\n")
    return p.transform(arr), p.transform(test)
