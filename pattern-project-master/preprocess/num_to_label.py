from sklearn.preprocessing import LabelEncoder

def num_to_label(x):
    s = set()
    for i in x:
        for j in i:
            s.add(j)

    s = list(s)
    le = LabelEncoder()
    le.fit(s)
    return le
