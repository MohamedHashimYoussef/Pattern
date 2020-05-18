from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from configs import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np
from sklearn.neighbors import KNeighborsClassifier



def ada_boost(x, y, mod="dt"):
    if mod == 'dt':
        clf = DecisionTreeClassifier(random_state=randomstate, max_depth=m_depth)
    elif mod == 'rf':
        clf = RandomForestClassifier(random_state=randomstate, max_depth=m_depth)
    elif mod == 'svm':
        clf = SVC(random_state=randomstate)
    elif mod == 'log':
        clf = LogisticRegression(random_state=randomstate)
    elif mod == 'knn':
        clf = KNeighborsClassifier(n_neighbors=3)
    else:
        clf = DecisionTreeClassifier(random_state=randomstate, max_depth=3)

    estimator = 200
    btd = AdaBoostClassifier(clf,
                             algorithm="SAMME",
                             n_estimators=estimator, random_state=randomstate
                             )
    m = -100
    crf = None
    kf = KFold(n_splits=folds)
    fbtd = btd.fit(x, y)

    return fbtd



