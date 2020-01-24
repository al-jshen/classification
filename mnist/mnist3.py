from sklearn.ensemble import RandomForestClassifier
import csv
import numpy as np

def read_dat(filepath, outlist):
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            outlist.append(list(map(int, row)))
training = []
read_dat('/Users/js/Desktop/mnist/train.csv', training)
training = np.array(training)
labels = training[:,0]
features = training[:,1:]

testing = []
read_dat('/Users/js/Desktop/mnist/test.csv', testing)

clf = RandomForestClassifier(n_estimators=100)
clf.fit(features, labels)

clf.predict(testing)


