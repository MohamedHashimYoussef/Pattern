from sklearn.naive_bayes import GaussianNB

def naiive(x , y):
    clf = GaussianNB()
    clf.fit(x,y)
    return clf