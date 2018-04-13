from random import seed
from random import randrange
from csv import reader
import pydot
MAX_VAL = 999999999


class DecisionTree():

    def __init__(self, max_depth=None, min_size=10, n_features=None):
        self.root = None
        self.max_depth = max_depth
        self.min_size = min_size
        self.n_features = n_features

    def _test_split(self, index, value, X, y):
        node = {
            'left': {'X': list(), 'y': list()},
            'right': {'X': list(), 'y': list()}
            }
        for i in range(len(X)):
            row = X[i]
            if row[index] < value:
                node['left']['y'].append(y[i])
                node['left']['X'].append(row)
            else:
                node['right']['y'].append(y[i])
                node['right']['X'].append(row)
        return node

    def _gini_index(self, groups, class_values):
        gini = 0.0
        for class_value in class_values:
            for orientation in groups:
                group = groups[orientation]
                size = float(len(group['y']))
                if size > 0:
                    proportion = group['y'].count(class_value) / size
                    gini += (proportion * (1.0 - proportion))
        return gini

    def _get_split(self, X, Y):
        class_values = list(set(y for y in Y))
        b_index, b_value, b_score, b_groups = MAX_VAL, MAX_VAL, MAX_VAL, None

        features = list()
        while len(features) < self.n_features:
            index = randrange(len(X[0]))
            if index not in features:
                features.append(index)

        for index in features:
            for row in X:
                groups = self._test_split(index, row[index], X, Y)
                gini = self._gini_index(groups, class_values)
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = \
                        index, row[index], gini, groups
        return {'index': b_index, 'value': b_value, 'groups': b_groups}

    def _to_leaf(self, target):
        return max(set(target), key=target.count)

    def _split(self, node, depth):
        group = node['groups']
        del(node['groups'])

        if len(group['right']['y']) == 0 or len(group['left']['y']) == 0:
            joined = group['left']['y'] + group['right']['y']
            node['left'] = node['right'] = self._to_leaf(joined)
            return

        if self.max_depth is not None:
            if depth >= self.max_depth:
                node['left'] = self._to_leaf(group['left']['y'])
                node['right'] = self._to_leaf(group['right']['y'])
                return

        for orientation in group:
            if len(set(group[orientation]['y'])) == 1:
                print("already 1 class")
                node[orientation] = self._to_leaf(group[orientation]['y'])
            elif len(group[orientation]['y']) <= self.min_size:
                print("min size reqched")
                node[orientation] = self._to_leaf(group[orientation]['y'])
            else:
                node[orientation] = self._get_split(
                    X=group[orientation]['X'], Y=group[orientation]['y'])
                self._split(node[orientation], depth + 1)

    def fit(self, X, Y):
        if self.n_features is None or len(X[0]) < self.n_features:
            self.n_features = len(X[0])
        self.root = self._get_split(X, Y)
        self._split(self.root, 0)

    def predict(self, row):
        node = self.root
        while isinstance(node, dict):
            if row[node['index']] < node['value']:
                node = node['left']
            else:
                node = node['right']
        return node

    def _get_label(self, node, orientation=None):
        meaning = {'left': '<', 'right': '>='}
        label = 'id: {}\nindex: {}'.format(
            str(id(node)),
            node['index']
        )
        if orientation:
            if isinstance(node[orientation], dict):
                label = '{} {}\nid: {}\nindex: {}'.format(
                    meaning[orientation],
                    node['value'],
                    str(id(node[orientation])),
                    node[orientation]['index']
                )
            else:
                label = '{} {}\nid: {}\nPREDICT: {}'.format(
                    meaning[orientation],
                    node['value'],
                    str(id(node[orientation])),
                    node[orientation]
                )
        return label

    def _draw(self, graph, parent_label, node):
        if not isinstance(node, dict):
            return
        left_label = self._get_label(node, 'left')
        right_label = self._get_label(node, 'right')

        left_edge = pydot.Edge(parent_label, left_label)
        graph.add_edge(left_edge)
        right_edge = pydot.Edge(parent_label, right_label)
        graph.add_edge(right_edge)
        self._draw(graph, left_label, node['left'])
        self._draw(graph, right_label, node['right'])

    def visualize(self, filename):
        graph = pydot.Dot(graph_type='graph')
        root_label = self._get_label(self.root)
        self._draw(graph, root_label, self.root)
        graph.write_png(filename)


class RandomForest():

    def __init__(self, max_depth, min_size, sample_ratio, n_trees, n_features):
        self.trees = list()
        self.max_depth = max_depth
        self.min_size = min_size
        self.sample_ratio = sample_ratio
        self.n_trees = n_trees
        self.n_features = n_features

    def rand_sample(self, X, Y):
        indexes = list(range(len(Y)))
        n_sample = round(len(Y) * self.sample_ratio)
        if n_sample == len(y):
            return X, Y
        sample_index = random.sample(indexes, n_sample)
        X_sample, Y_sample = list(), list()
        for i in sample_index:
            X_sample.append(X[i])
            Y_sample.append(Y[i])
        return X_sample, Y_sample

    def fit(self, X, Y):
        for i in range(self.n_trees):
            X_sample, Y_sample = self.rand_sample(X, Y)
            dt = DecisionTree(
                max_depth=self.max_depth,
                min_size=self.min_size,
                n_features=self.n_features
            )
            dt.fit(X_sample, Y_sample)
            self.trees.append(dt)

    def predict(self, x):
        predictions = [tree.predict(x) for tree in self.trees]
        return max(set(predictions), key=predictions.count)



########### EXAMPLE USAGE ############################

def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            temp = [float(r) for r in row]
            dataset.append(temp)
        return dataset

dataset = load_csv('sample_data.csv')


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

# dt = RandomForest(max_depth=1, min_size=1, n_features=5, n_trees=10, sample_ratio=0.8)
dt = DecisionTree(max_depth=None, min_size=1)
dt.fit(X_train, y_train)
dt.visualize('example.png')
predicted = [dt.predict(x) for x in X_test]

print("actual :", y_test)
print("predicted :", predicted)
print(accuracy_metric(y_test, predicted))
