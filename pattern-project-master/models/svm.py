from sklearn.svm import SVC
from configs import *


def svm(x, y):
    clf = SVC(random_state=randomstate)
    clf.fit(x, y)
    return clf
