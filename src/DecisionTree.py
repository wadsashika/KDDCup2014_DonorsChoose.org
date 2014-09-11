__author__ = 'dulanga'

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.externals.six import StringIO
from sklearn import tree

projects = pd.read_csv('/home/dulanga/Documents/Python/projects.csv').sort('projectid')

dates = np.array(projects.date_posted)
train_idx = np.where(dates < '2014-01-01')[0]
test_idx = np.where(dates >= '2014-01-01')[0]

projects = projects.fillna(method='pad')

projects = np.array(projects)

for i in range(0, projects.shape[1]):
    le = LabelEncoder()
    projects[:, i] = le.fit_transform(projects[:, i])
projects = projects.astype(float)

train = projects[train_idx]
test = projects[test_idx]

outcomes = pd.read_csv('/home/dulanga/Documents/Python/outcomes.csv').sort('projectid')
outcomes = np.array(outcomes.is_exciting)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train, outcomes == 't')

with open("iris.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)

print(clf)
