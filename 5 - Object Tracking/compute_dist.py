import numpy as np
from scipy.spatial import distance as dist

X = np.array([[1,2],[3,4]])
Y = np.array([[3,3],[1,5]])

D = dist.cdist(X, Y)
print(D)

rows = D.min(axis=1).argsort()
print(rows)

cols = D.argmin(axis=1)[rows]
print(cols)
