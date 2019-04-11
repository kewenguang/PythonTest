'''
Created on 2019年3月16日

@author: kewenguang
'''
from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt 

x = [2.273, 27.89, 30.519, 62.049, 29.263, 62.657, 75.735, 24.344, 17.667, 68.816, 69.076, 85.691] 
y = [68.367, 83.127, 61.07, 69.343, 68.748, 90.094, 62.761, 43.816, 86.765, 76.874, 57.829, 88.114] 
plt.plot(x, y, 'b.') 
plt.show() 

points = [[i,j] for i,j in zip(x,y)]#Python递推式，将x和y中的数据依次选出构成点集 
y_pred = KMeans(n_clusters=2).fit_predict(points)#将数据聚为2类 
print('聚类结果：', y_pred)#打印聚类的结果 
plt.scatter(x, y, c=y_pred, marker='*') 
plt.show()
