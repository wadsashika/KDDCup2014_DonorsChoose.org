__author__ = 'dulanga'

import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import ensemble


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
    wordCount.append(((len(str(essays[i]).split()) / 20) + 1) * 20)

del essays
projects['essay_length'] = wordCount
del wordCount

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

print(projects.shape[1])

for i in range(0, projects.shape[1]):
    le = LabelEncoder()
    projects[:, i] = le.fit_transform(projects[:, i])
projects = projects.astype(float)

print("come1")
ohe = OneHotEncoder()
projects = ohe.fit_transform(projects)


train = projects[train_idx]
test = projects[test_idx]
del projects
print("come 2")

outcomes = pd.read_csv('/home/dulanga/Documents/Python/outcomes.csv').sort('projectid')
outcomes = np.array(outcomes.is_exciting)

params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 1,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(train, outcomes)

feature_importance = clf.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
pl.subplot(1, 2, 1)
pl.barh(pos, feature_importance[sorted_idx], align='center')
pl.yticks(pos, train.feature_names[sorted_idx])
pl.xlabel('Relative Importance')
pl.title('Variable Importance')
pl.show()