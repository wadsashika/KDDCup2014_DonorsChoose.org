__author__ = 'dulanga'

import numpy as np

data1 = [['dsad', 'dafaef', 'asdaf'],
         ['sdfsdf', 'afaf', 'aff']]
data2 = [['ggerg'],
         ['sdfsdf']]

data11 = np.array(data1)
data22 = np.array(data2)
# print(data11)
# print(data22)
print(np.append(data11, data22,1))
