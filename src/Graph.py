__author__ = 'dulanga'

# predictions_v6 is the best for now

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# <codecell>


projects = pd.read_csv('/home/dulanga/Documents/Python/projects.csv').sort('projectid')

# <codecell>

dates = np.array(projects.date_posted)
train_idx = np.where(dates < '2014-01-01')[0]
test_idx = np.where(dates >= '2014-01-01')[0]
#
# # <codecell>
#
projects = projects.fillna(method='pad')

projectCatogorialColumns = ['teacher_acctid', 'schoolid', 'school_ncesid', 'school_city', 'school_state',
                            'school_district', 'total_price_including_optional_support',
                            'school_county', 'school_charter', 'school_magnet',
                            'school_year_round', 'school_nlns', 'school_kipp', 'school_charter_ready_promise',
                            'teacher_prefix', 'teacher_teach_for_america', 'teacher_ny_teaching_fellow',
                            'primary_focus_subject', 'primary_focus_area',
                            'secondary_focus_subject', 'secondary_focus_area', 'resource_type', 'poverty_level',
                            'grade_level', 'fulfillment_labor_materials',
                            'eligible_double_your_impact_match', 'eligible_almost_home_match']

# <codecell>

projects = np.array(projects[projectCatogorialColumns])

print(projects.shape[1])
# <codecell>

for i in range(0, projects.shape[1]):
    le = LabelEncoder()
    projects[:, i] = le.fit_transform(projects[:, i])
projects = projects.astype(float)

#
train = projects[train_idx]
test = projects[test_idx]
train2 = train[6:]

outcomes = pd.read_csv('/home/dulanga/Documents/Python/outcomes.csv').sort('projectid')
outcomes = np.array(outcomes.is_exciting)
outcomes2 = []
for i in range(0, len(outcomes) - 6):
    if outcomes[i] == 't':
        outcomes2.append(1)
    else:
        outcomes2.append(0)

outcomes2 = np.array(outcomes2)

print(len(train2))
print(len(outcomes2))
plt.plot(outcomes2, train2, 'bo')
plt.show()

