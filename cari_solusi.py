from RandomForest import RandomForestClassifier as manualRF
from sklearn.ensemble import RandomForestClassifier as sklearnRF
import pandas as pd
import numpy as np
import joblib
#import matplotlib.pyplot as plt
#import seaborn as sns

from sklearn.model_selection import train_test_split


one_hot_encoded = pd.read_csv('combats_data.csv')
#print(one_hot_encoded.describe())
# cols = [c for c in one_hot_encoded.columns if 'Type' not in c]
# for i in range(len(cols)):
#     print(str(i), cols[i])
# one_hot_encoded = one_hot_encoded[cols]
X = one_hot_encoded.drop(['First_win'], axis=1)
Y = one_hot_encoded['First_win']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

np_X_train = np.array(X_train)
np_y_train = np.array(y_train)
np_X_test = np.array(X_test)
np_y_test = np.array(y_test)

import time
manual_start = time.time()
clf_manual = manualRF(n_trees=11, max_depth=11)
clf_manual.fit(np_X_train, np_y_train)
manual_end = manual_start - time.time()
#clf_manual = joblib.load('trained_5_10')

sklearn_start = time.time()
clf_sklearn = sklearnRF(n_estimators=11, max_depth=11).fit(np_X_train, np_y_train)
clf_sklearn.fit(np_X_train, np_y_train)
sklearn_end = sklearn_start - time.time()


sklearn_predicted = clf_sklearn.predict(np_X_test)
manual_predicted = list()
try:
    joblib.dump(clf_manual, 'trained_10_10')
except Exception as e:
    print("gagal dump")
for i in range(len(np_y_test)):
    manual_predicted.append(clf_manual.predict(np_X_test[i]))
manual_predicted = np.array(manual_predicted)

from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import accuracy_score

tn_sklearn, fp_sklearn, fn_sklearn, tp_sklearn = confusion_matrix(y_test, sklearn_predicted).ravel()
acc_sklearn = accuracy_score(y_test, sklearn_predicted)
f1_score_sklearn = f1_score(y_test, sklearn_predicted)


tn_manual, fp_manual, fn_manual, tp_manual = confusion_matrix(y_test, manual_predicted).ravel()
acc_manual = accuracy_score(y_test, manual_predicted)
f1_score_manual = f1_score(y_test, manual_predicted)

print("SKLEARN:")
print("TP: {}".format(str(tp_sklearn)))
print("TN: {}".format(str(tn_sklearn)))
print("FP: {}".format(str(fp_sklearn)))
print("FN: {}".format(str(fn_sklearn)))
print("Accuracy: {}".format(acc_sklearn))
print("F1 Score: {}".format(f1_score_sklearn))
print("elapsed time: {}".format(str(sklearn_end)))

print("Manual:")
print("TP: {}".format(str(tp_manual)))
print("TN: {}".format(str(tn_manual)))
print("FP: {}".format(str(fp_manual)))
print("FN: {}".format(str(fn_manual)))
print("Accuracy: {}".format(acc_manual))
print("F1 Score: {}".format(f1_score_manual))
print("elapsed time: {}".format(manual_end))



from visualizer import visualize
for i in range(len(clf_manual.trees)):
    visualize(clf_manual.trees[i],"tree_{}.png".format(str(i)))
