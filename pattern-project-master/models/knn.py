from sklearn.neighbors import KNeighborsClassifier


def knn(x, y):
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(x, y)

    return clf

