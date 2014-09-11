# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime

# <codecell>

startTime = datetime.now()

# <codecell>

#donations = pd.read_csv('../data/donations.csv').sort('projectid')
projects = pd.read_csv('/home/dulanga/Documents/Python/projects.csv').sort('projectid')
outcomes = pd.read_csv('/home/dulanga/Documents/Python/outcomes.csv').sort('projectid')
#resources = pd.read_csv('../data/resources.csv').sort('projectid')
sample = pd.read_csv('/home/dulanga/Documents/Python/sampleSubmission.csv').sort('projectid')
#essays = pd.read_csv('../data/essays.csv').sort('projectid')

# <codecell>

dates = np.array(projects.date_posted)
train_idx = np.where(dates < '2014-01-01')[0]
test_idx = np.where(dates >= '2014-01-01')[0]

# <codecell>

projects = projects.fillna(method='pad')

# <codecell>

outcomes = np.array(outcomes.is_exciting)

# <codecell>

projectCatogorialColumns = ['teacher_acctid', 'schoolid', 'school_ncesid','school_city', 'school_state', 'school_zip', 'school_metro', 'school_district',
                            'school_county', 'school_charter', 'school_magnet',
                            'school_year_round', 'school_nlns', 'school_kipp', 'school_charter_ready_promise',
                            'teacher_prefix', 'teacher_teach_for_america', 'teacher_ny_teaching_fellow',
                            'primary_focus_subject', 'primary_focus_area',
                            'secondary_focus_subject', 'secondary_focus_area', 'resource_type', 'poverty_level',
                            'grade_level','fulfillment_labor_materials',
                            'students_reached', 'eligible_double_your_impact_match', 'eligible_almost_home_match']
latitudeLongitudeColumns = ['school_latitude', 'school_longitude']

# <codecell>

latitudeLongitude = np.array(projects[latitudeLongitudeColumns])
latitudeLongitude = np.ceil(latitudeLongitude)
latitudeLongitude[:,0] = 180*latitudeLongitude[:,0]+latitudeLongitude[:,1]

# <codecell>

data = np.array(projects[projectCatogorialColumns])
data = np.column_stack((data,latitudeLongitude[:,0]))
del projects, latitudeLongitude

# <codecell>

for i in range(0, data.shape[1]):
    le = LabelEncoder()
    data[:,i] = le.fit_transform(data[:,i])
data = data.astype(float)

# <codecell>

ohe = OneHotEncoder()
data = ohe.fit_transform(data)

# <codecell>

train = data[train_idx]
test = data[test_idx]
del data

# <codecell>

model = SGDClassifier(alpha = 0.001, loss = 'modified_huber', penalty = 'l2', n_iter = 1000, n_jobs = -1)
model.fit(train, outcomes=='t')

# <codecell>

preds = model.predict_proba(test)[:,1]

# <codecell>

endTime = datetime.now()

# <codecell>

sample['is_exciting'] = preds
sample.to_csv('predictions.csv', index = False)

# <codecell>

print endTime - startTime

