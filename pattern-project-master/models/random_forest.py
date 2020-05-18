from sklearn.ensemble import RandomForestClassifier
from configs import *


def random_forest(x, y):
    clf = RandomForestClassifier(random_state=randomstate, max_depth=m_depth)
    clf.fit(x, y)
    return clf
