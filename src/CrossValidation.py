# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from __future__ import division
import pandas as pd
import numpy as np

projects = pd.read_csv('/home/dulanga/Documents/Python/projects.csv').sort('projectid')

dates = np.array(projects.date_posted)
train_idx = np.where(dates < '2014-01-01')[0]
test_idx = np.where(dates >= '2014-01-01')[0]

projects = projects.fillna(method='pad')

projects['month'] = ''
projects['total_price_including'] = 0
projects['students_reached_grouped'] = 0
for i in range(0, projects.shape[0]):
    projects['month'][i] = projects.date_posted[i][5:7]

    totalPrice = projects.total_price_including_optional_support[i]
    if (totalPrice < 250):
        projects.total_price_including[i] = 0
    elif ((totalPrice >= 250) & (totalPrice < 400)):
        projects.total_price_including[i] = 1
    elif ((totalPrice >= 400) & (totalPrice < 600)):
        projects.total_price_including[i] = 2
    elif ((totalPrice >= 600) & (totalPrice < 10000)):
        projects.total_price_including[i] = 3
    elif ((totalPrice >= 10000) & (totalPrice < 100000)):
        projects.total_price_including[i] = 4
    else:
        projects.total_price_including[i] = 5

    studentNo = int(projects.students_reached[i])
    if (studentNo == 0):
        projects.students_reached_grouped[i] = 0
    elif (studentNo < 100):
        projects.students_reached_grouped[i] = (studentNo / 5) + 1
    elif (studentNo <= 500):
        projects.students_reached_grouped[i] = 100
    else:
        projects.students_reached_grouped[i] = 1000

essays = pd.read_csv('/home/dulanga/Documents/Python/essays.csv').sort('projectid')
essays = np.array(essays.essay)
wordCount = []

for i in range(0, len(essays)):
    wordCount.append(((len(str(essays[i]).split()) / 10) + 1))

del essays
projects['essay_length'] = wordCount
del wordCount


# <codecell>

outcomes = pd.read_csv('/home/dulanga/Documents/Python/outcomes.csv').sort('projectid')
outcomes = outcomes['is_exciting']
y = np.where(outcomes == 't', 1, 0)

projectCatogorialColumns = ['teacher_acctid', 'schoolid', 'school_city', 'school_state', 'school_district',
                            'school_metro', 'school_county', 'school_charter', 'school_magnet',
                            'school_year_round', 'school_nlns', 'school_kipp', 'school_charter_ready_promise',
                            'teacher_prefix', 'teacher_teach_for_america', 'teacher_ny_teaching_fellow',
                            'primary_focus_subject', 'primary_focus_area',
                            'secondary_focus_subject', 'secondary_focus_area', 'resource_type', 'poverty_level',
                            'grade_level', 'fulfillment_labor_materials', 'total_price_including',
                            'students_reached_grouped', 'eligible_double_your_impact_match',
                            'eligible_almost_home_match', 'essay_length', 'month']

projects = np.array(projects[projectCatogorialColumns])

from sklearn.preprocessing import LabelEncoder

for i in range(0, projects.shape[1]):
    le = LabelEncoder()
    projects[:, i] = le.fit_transform(projects[:, i])
projects = projects.astype(float)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(projects)

print "Feature space holds %d observations and %d features" % projects.shape
print "Unique target labels:", np.unique(y)


# <codecell>

print y

# <codecell>

from sklearn.cross_validation import KFold


def run_cv(X, y, clf_class, **kwargs):
    # Construct a kfolds object
    kf = KFold(len(y), n_folds=3, shuffle=True)
    y_pred = y.copy()

    # Iterate through folds
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        # Initialize a classifier with key word arguments
        clf = clf_class(**kwargs)
        clf.fit(X_train, y_train)
        y_pred[test_index] = clf.predict(X_test)
    return y_pred


from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import SGDClassifier as SGD


def accuracy(y_true, y_pred):
    # NumPy interpretes True and False as 1. and 0.
    return np.mean(y_true == y_pred)


X1 = X[train_idx]
X2 = X[test_idx]
del X

# print "Support vector machines:"
# print "%.3f" % accuracy(y, run_cv(X1,y,SVC))
# print "Random forest:"
# print "%.3f" % accuracy(y, run_cv(X1,y,RF))
# print "K-nearest-neighbors:"
# print "%.3f" % accuracy(y, run_cv(X1,y,KNN))
print "Logistic Regression:"
print "%f" % accuracy(y, run_cv(X1,y,LR))
# print "SGD:"
# print "%.3f" % accuracy(y, run_cv(X1,y,SGD))

# <codecell>

# LR 0.941
# RF 0.940
# SGD 0.941
#KNN 0.940


