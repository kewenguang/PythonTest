'''
Created on 2019年1月13日

@author: kewenguang
'''
import numpy as np  
import matplotlib.pyplot as plt 
from array import array
#原始数据 假设关系为 y=3+x+2z  ==== y=a+bx+cz
tmpx=[1,2,3,4,5,6]
tmpy=[1,1,2,2,3,3]
tmpz=[6,7,10,11,14,15]

x=np.mat(tmpx)
z=np.mat(tmpy)
y=np.mat(tmpz)
#print(x)
y=y.T
a=0
b=0
c=0
canshu=np.mat([a,b,c])
canshu=canshu.T

print(canshu)
bianliang=np.column_stack(([1,1,1,1,1,1],x.T,z.T))
#print(bianliang[:,0])
#print(bianliang[:,1])
print(bianliang)
#学习速率
canshu=(bianliang.T*bianliang).I*bianliang.T*y
print(canshu)


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
 
fig = plt.figure()
#ax = fig.gca(projection='3d')

#下面是画点的
ax = Axes3D(fig)
ax.scatter(list(tmpx),list(tmpy),list(tmpz))



# Make data.
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
theta = [canshu[0,0],canshu[1,0],canshu[2,0] ]
Z = X * theta[0] + Y * theta[1] + theta[2]
print(theta[0])
print(X)
print(Y)
print(Z)

#X, Y = np.meshgrid(X, Y)
#R = np.sqrt(X ** 2 + Y ** 2)
#Z = np.sin(R)
#print(X)
#rint(Y)
#rint(Z)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap='rainbow',linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-15.01, 15.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
 
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
 
plt.show()