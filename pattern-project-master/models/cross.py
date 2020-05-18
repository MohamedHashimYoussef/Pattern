from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from configs import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np


def cross(x, y, mod="dt"):
    if mod == 'dt':
        clf = DecisionTreeClassifier(random_state=randomstate, max_depth=m_depth)
    elif mod == 'rf':
        clf = RandomForestClassifier(random_state=randomstate, max_depth=m_depth)
    elif mod == 'svm':
        clf = SVC(random_state=randomstate)
    elif mod == 'log':
        clf = LogisticRegression(random_state=randomstate)
    else:
        clf = DecisionTreeClassifier(random_state=randomstate, max_depth=m_depth)

    kf = KFold(n_splits=5)

    m = -100
    crf = None

    for train, test in kf.split(x):

        X_train, X_test, y_train, y_test = np.array(x)[train], np.array(x)[test], np.array(y)[train], np.array(y)[test]
        clf.fit(X_train, y_train)
        pr = clf.predict(X_test)

        acc = accuracy_score(y_test, pr)

        if acc > m:
            m = acc
            crf = clf
            print(f"=>>>>>>>>>>> {acc}")

    return crf
