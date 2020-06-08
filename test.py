from scipy.sparse import csr_matrix
import numpy as np
import random

row = np.array([0, 0, 1, 2, 2, 2])
col = np.array([0, 2, 2, 0, 1, 2])
data = np.array([1, 2, 3, 4, 5, 6])
mtr = csr_matrix((data, (row, col)))

print(mtr.todense())

rows, cols = mtr.nonzero()
data = mtr[rows, cols]

print(np.array(data).flatten())

dict = {0: 3}
dict['abc']  = 1234
dict['adbc']  = 1234
abc = dict['abc']
print(len(dict))
for i in dict:
    print(i)

