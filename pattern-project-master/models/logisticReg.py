from sklearn.linear_model import LogisticRegression
from configs import *


def log_reg(x, y):
    clf = LogisticRegression(random_state=randomstate)
    clf.fit(x, y)
    return clf


