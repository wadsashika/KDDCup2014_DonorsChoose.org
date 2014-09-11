__author__ = 'dulanga'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


projects = pd.read_csv('/home/dulanga/Documents/Python/projects.csv').sort('projectid')


dates = np.array(projects.date_posted)
train_idx = np.where(dates < '2014-01-01')[0]
test_idx = np.where(dates >= '2014-01-01')[0]

projects = projects.fillna(method='pad')

projects['total_price_including'] = 0
projects['students_reached_grouped'] = 0
for i in range(0, projects.shape[0]):

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

col = ['projectid','total_price_including', 'students_reached_grouped']

projects2 = np.array(projects[col])
# projects2 = projects2.astype(float)

print(train_idx)

train = projects2[train_idx]

print("come 1")
print(train)

train = train.tolist()

train2 = pd.DataFrame(train)
print(train2)

train2.to_csv('training_set.csv', index=False)