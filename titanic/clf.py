import csv
import numpy as np
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

def impt(file):
    with open(file, 'r') as f:
        reader = csv.reader(f)
        dataset = list(reader)
        data_columns = dataset[0]
        data_values = dataset[1:]

    '''
    Passenger ID, Pclass, Name, Sex, Age, SibSp, Parch, Ticket,
    Fare, Cabin, Embarked
    '''

    # extract labels from data (survival status)
    labels = [i[1] for i in data_values]

    '''
    remove useless columns (name, id, ticket #, cabin (lots of missing data)
    done in decreasing index so that it doesnt mess with the following statements
    '''
    if file == 'train.csv':
        for j in data_values:
            del j[10]
            del j[8]
            del j[3]
            del j[1]
            del j[0]
    elif file == 'test.csv':
        for j in data_values:
            del j[9]
            del j[7]
            del j[2]
            del j[0]

    # pclass, sex, age, sibsp, parch, fare, embarked

    '''
    get rid of entries with missing data for both features and labels
    doesnt clean all data after running once so uh i run multiple passes
    '''

    for i in range(10):
        indices = [data_values.index(i) for i in data_values if '' in i]
        indices.sort(reverse=1)
        for m in indices:
            del data_values[m]
            if file == 'train.csv':
                del labels[m]

    # consolidate sibsp and parch (# of family members)
    for k in data_values:
        fam = int(k[3]) + int(k[4])
        k.append(fam)
        del k[4]
        del k[3]


    # pclass, sex, age, fare, embarked, fam

    # turn sex and embarkment port into numbers
    for l in data_values:
        if l[1] == 'male':
            l[1] = 0
        elif l[1] == 'female':
            l[1] = 1

        if l[4] == 'C':
            l[4] = 1
        elif l[4] == 'Q':
            l[4] = 2
        elif l[4] == 'S':
            l[4] = 3

    # turn all data types into floats
    features = np.array([list(map(float, o)) for o in data_values])
    if file == 'train.csv':
        labels = np.array(labels)

    if file == 'train.csv':
        return features, labels
    elif file == 'test.csv':
        return features

#X_train, Y_train = impt('train.csv')
#X_test = impt('test.csv')


features, labels = impt('train.csv')
X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.3)

'''

idk using gridsearchcv just wasted a huge amount of time
would take like 5 mins to fit all the svm stuff
but would just end up with around 80% accuracy
which is the same as not using it
so now i dont use it but its here just for info


svm = SVC()
nb = GaussianNB()
forest = RandomForestClassifier()

svm_parameters = {'C':(1, 10, 100, 1000),
                  'kernel':('linear', 'rbf'),
                  'gamma':(0.00001, 0.0001, 0.001, 0.01, 0.1, 1)}

forest_parameters = {'criterion':('gini', 'entropy'),
              'n_estimators':(10, 50, 100, 1000),
              'min_samples_split':(2, 3, 4, 5)}

clf = GridSearchCV(svm, svm_parameters, cv=StratifiedKFold(), n_jobs=-1, verbose=1)
'''
#clf = RandomForestClassifier(criterion='entropy', n_estimators=100, min_samples_split=4)
clf = SVC(C=100, kernel='linear', gamma=0.00001)
clf.fit(X_train, Y_train)
print(clf.score(X_test, Y_test))

#preds = list(clf.predict(X_test))
#print(preds.count('1')/len(preds))

