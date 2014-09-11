# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pylab as p

# <codecell>

projects = pd.read_csv('/home/dulanga/Documents/Python/projects.csv')
outcome = pd.read_csv('/home/dulanga/Documents/Python/outcomes.csv')

# <codecell>

projects = projects.merge(outcome, how = 'inner')

# <codecell>

projects = projects.sort('date_posted')

# <codecell>

le = LabelEncoder()

# <codecell>

for i in range(0,projects.shape[0]):
    projects.date_posted[i] = projects.date_posted[i][0:7]

# <codecell>

projects['date_posted'] = le.fit_transform(projects.date_posted)

# <codecell>

projects = projects[['date_posted','is_exciting']]

# <codecell>

tot = projects[projects.date_posted == 0]
true = [float(tot[tot.is_exciting == 't'].count()[0])/float(tot.count()[0])]
for i in range(1,projects.date_posted.max()+1):
    tot = projects[projects.date_posted == i]
    true.append(float(tot[tot.is_exciting == 't'].count()[0])/float(tot.count()[0]))

# <codecell>

p.plot(true)

# <codecell>

p.show()

