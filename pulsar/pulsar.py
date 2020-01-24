import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

data = np.loadtxt('pulsar_stars.csv', skiprows=1, delimiter=',')
x = data[:,:-1]
y = data[:,-1]

def pred(model, x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
    model.fit(x_train, y_train)
    return clf.score(x_test, y_test)

clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)

print(pred(clf, x, y))

pca = PCA(n_components=2)
pca.fit(x)
print(pca.explained_variance_)

# no point in doing this here
#x2 = pca.fit_transform(x)
#print(pred(clf, x2, y))

plt.scatter(x2[:,0], x2[:,1], c=y)
plt.show()




