import numpy as np
import matplotlib.pyplot as mp
from sklearn.linear_model import LogisticRegression
 
x = np.array([
    [4, 7],
    [3.5, 8],
    [3.1, 6.2],
    [0.5, 1],
    [1, 2],
    [1.2, 1.9],
    [6, 2],
    [5.7, 1.5],
    [5.4, 2.2]
])
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
model = LogisticRegression(solver='liblinear', C=100)#C和正则化有关 较小的C值代表较强的正则化    少量数据用liblinear
model.fit(x, y)
l, r, h = x[:, 0].min() - 1, x[:, 0].max() + 1, 0.05
b, t, v = x[:, 1].min() - 1, x[:, 1].max() + 1, 0.05
grid_x = np.meshgrid(np.arange(l, r, h), np.arange(b, t, v))
flat_x = np.c_[grid_x[0].ravel(), grid_x[1].ravel()]
flat_y = model.predict(flat_x)
grid_y = flat_y.reshape(grid_x[0].shape)
 
mp.figure('Logistic Classification', facecolor='lightgray')
mp.title('Logistic Classification', fontsize=12)
mp.xlabel('x', fontsize=12)
mp.ylabel('y', fontsize=12)
mp.tick_params(labelsize=10)
mp.pcolormesh(grid_x[0], grid_x[1], grid_y, cmap='gray')
mp.scatter(x[:, 0], x[:, 1], c=y, cmap='brg', s=60)
mp.show()