from random import seed
from random import randrange
import random
from multiprocessing import Process, Queue
from threading import Thread
MAX_VAL = 999999999

# Muhammad - 1506735641 - A
# Nur Intan - 1506689093  - A
# Ahmad Elang - 1506689105 - A


class DecisionTree():

    def __init__(self, max_depth=None, min_size=1, n_features=None):
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
                else:
                    gini += MAX_VAL
        return gini

    def _get_split(self, X, Y):
        class_values = list(set(Y))
        b_index, b_value, b_score, b_groups = MAX_VAL, MAX_VAL, MAX_VAL, None
        features = random.sample(range(len(X[0])), self.n_features)
        for index in features:
            seen = {}
            seen.clear()
            for row in X:
                # SKIP YANG SUDAH PERNAH DILIAT
                if not seen.get(row[index], False):
                    seen.update({row[index]: True})
                    groups = self._test_split(index, row[index], X, Y)
                    gini = self._gini_index(groups, class_values)
                    if gini < b_score:
                        b_index, b_value, b_score, b_groups = \
                            index, row[index], gini, groups

        return {'index': b_index, 'value': b_value, 'groups': b_groups}

    def _to_leaf(self, target):
        return max(set(target), key=target.count)

    def _get_split_thread(self, node, orientation, group):
        node[orientation] = self._get_split(
            X=group[orientation]['X'], Y=group[orientation]['y'])

    def _split(self, node, depth):
        group = node['groups']
        del(node['groups'])
        if group is None:
            return

        if len(group['right']['y']) == 0 or len(group['left']['y']) == 0:
            joined = group['left']['y'] + group['right']['y']
            node['left'] = node['right'] = self._to_leaf(joined)
            return

        if self.max_depth is not None:
            if depth >= self.max_depth:
                node['left'] = self._to_leaf(group['left']['y'])
                node['right'] = self._to_leaf(group['right']['y'])
                return

        threads = {}
        for orientation in group:
            if len(set(group[orientation]['y'])) == 1:
                node[orientation] = self._to_leaf(group[orientation]['y'])
            elif len(group[orientation]['y']) <= self.min_size:
                node[orientation] = self._to_leaf(group[orientation]['y'])
            else:
                threads[orientation] = Thread(
                    target=self._get_split_thread,
                    args=(node, orientation, group)
                )
                threads[orientation].start()

        for orientation in threads:
            threads[orientation].join()
            self._split(node[orientation], depth + 1)

    def fit(self, X, Y):
        print("start training DT")
        if self.n_features is None or len(X[0]) < self.n_features:
            self.n_features = len(X[0])
        self.root = self._get_split(X, Y)
        self._split(self.root, 0)
        print("done training DT")

    def predict(self, row):
        node = self.root
        while isinstance(node, dict):
            if row[node['index']] < node['value']:
                node = node['left']
            else:
                node = node['right']
        return node


class RandomForestClassifier():

    def __init__(self, max_depth=None, min_size=1, sample_ratio=1.0, n_trees=10, n_features=None):
        self.trees = list()
        self.max_depth = max_depth
        self.min_size = min_size
        self.sample_ratio = sample_ratio
        self.n_trees = n_trees
        self.n_features = n_features

    def rand_sample(self, X, Y):
        indexes = list(range(len(Y)))
        n_sample = round(len(Y) * self.sample_ratio)
        if n_sample == len(Y):
            return X, Y
        sample_index = random.sample(indexes, n_sample)
        X_sample, Y_sample = list(), list()
        for i in sample_index:
            X_sample.append(X[i])
            Y_sample.append(Y[i])
        return X_sample, Y_sample

    def _fit_tree(self, X, Y, q):
        X_sample, Y_sample = self.rand_sample(X, Y)
        dt = DecisionTree(
            max_depth=self.max_depth,
            min_size=self.min_size,
            n_features=self.n_features
        )
        dt.fit(X_sample, Y_sample)
        q.put(dt)

    def fit(self, X, Y):
        q = Queue()
        processes = []
        print("start training")
        for i in range(self.n_trees):
            p = Process(target=self._fit_tree, args=(X, Y, q))
            p.start()
            processes.append(p)
        print("training...")
        for proc in processes:
            proc.join()
        print("done training")
        q.put('STOP')
        for tree in iter(q.get, 'STOP'):
            if tree != 'STOP':
                self.trees.append(tree)

    def _predict_one(self, x):
        predictions = [tree.predict(x) for tree in self.trees]
        return max(set(predictions), key=predictions.count)

    def predict(self, X):
        return [self._predict_one(x) for x in X]
