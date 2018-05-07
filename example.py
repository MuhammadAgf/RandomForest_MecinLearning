from RandomForest import RandomForestClassifier, DecisionTree
from csv import reader
#from visualizer import visualize

########### EXAMPLE USAGE ############################
label_map = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}

def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row: break
            X = row[:-1]
            y = row[-1]
            if not row:
                continue
            temp = [float(r) for r in X]
            temp.append(label_map[y])
            dataset.append(temp)
        return dataset

dataset = load_csv('iris.csv')


import random
random.shuffle(dataset)
X = [y[0:4] for y in dataset]
y = [y[-1] for y in dataset]
offset = int(len(y) * 0.2)
y_train = y[offset:]
X_train = X[offset:]
y_test = y[:offset]
X_test = X[:offset]
print(len(X_test))

max_depth = 10
min_size = 1
sample_size = 1.0

def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

#clf = RandomForestClassifier(max_depth=None, min_size=1, n_features=None, n_trees=100, sample_ratio=0.8)
#clf = RandomForestClassifier()
#clf.fit(X_train, y_train)
clf = DecisionTree()
clf.fit(X_train, y_train)
from pprint import pprint
pprint(clf.root)
#visualize(clf,'contoh.png')
predicted = [clf.predict(x) for x in X_test]

print("actual:", y_test)
print("predicted:", predicted)
print("accuracy:", accuracy_metric(y_test, predicted))
