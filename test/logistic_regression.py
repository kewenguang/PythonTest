'''
Created on 2019年1月20日

@author: kewenguang
'''
from numpy import *
import matplotlib.pyplot as plt

# 梯度下降法
def gardDescent(dataMatIn,classLabels): 
    dataMatrix = mat(dataMatIn) 
    labelMatrix = mat(classLabels).T 
    m,n = shape(dataMatrix) # 得到数据规模 # 迭代步长 
    alpha = 0.01 # 迭代次数
    maxCycles = 5000 
    weights = ones((n,1)) # help(numpy.ones) # 设定初始参数，全为1 
    for k in range(maxCycles): 
        h = sigmoid(dataMatrix * weights) # sigmoid函数已定义 
        E = (h - labelMatrix) 
        weights = weights - alpha * dataMatrix.T * E 
    return weights

# 梯度上升算法
def stocGradAscent0(dataMatrix,classLabels): 
    dataMatrix = array(dataMatrix) 
    m,n = shape(dataMatrix) 
    alpha = 0.01 
    weights = ones(n) 
    for i in range(m): 
        h = sigmoid(sum(dataMatrix[i] * weights)) 
        E = classLabels[i] - h 
        weights = weights + alpha * E * dataMatrix[i] 
    return weights

# 改进的随机梯度上升算法 
def stocGradAscent1(dataMatrix,classLabels): 
    dataMatrix = array(dataMatrix) 
    m,n = shape(dataMatrix) 
    alpha = 0.01 
    weights = ones(n) 
    # 在所有样本点上迭代500次 
    for j in range(500): 
        for i in range(m): 
            h = sigmoid(sum(dataMatrix[i] * weights)) 
            E = classLabels[i] - h 
            weights = weights + alpha * E * dataMatrix[i] 
    return weights

def loadDataSet():
    dataMat = [] 
    labelMat = [] 
    fr = open('testSet.txt') # 逐行读入数据，然后strip去头去尾，用split分组 
    for line in fr.readlines(): 
        lineArr = line.strip().split('   ') 
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])]) 
        labelMat.append(int(lineArr[2])) 
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

def plotBestFit(weights_1,weights_2,weights_3):
    weights_1 = weights_1.getA()
    dataMat,labelMatrix = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n): 
        if int(labelMatrix[i]) == 1: 
            xcord1.append(dataArr[i,1]) 
            ycord1.append(dataArr[i,2]) 
        else: 
            xcord2.append(dataArr[i,1]) 
            ycord2.append(dataArr[i,2])
    fig = plt.figure(figsize=(14,6))
    
    
    #############################图1##################################
    ax = fig.add_subplot(221) # 画散点图，不同的样本点用不同颜色表示 
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='blue',)
    x = arange(-3.0,3.0,0.1)
    y_1 = (-weights_1[0]-weights_1[1]*x)/(weights_1[2])
    ax.plot(x,y_1,'k--',color = 'yellow',linewidth=2)
    plt.xlabel('Logistics Regression GradDescent')
    # 去掉坐标系右边和上边的边界，美观
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    
    #############################图2##################################
    ax = fig.add_subplot(222) 
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s') 
    ax.scatter(xcord2,ycord2,s=30,c='blue',) 
    x = arange(-3.0,3.0,0.1) 
    y_2 = (-weights_2[0]-weights_2[1]*x)/(weights_2[2]) 
    ax.plot(x,y_2,'k--',color = 'yellow',linewidth=2) 
    plt.xlabel('Logistics Regression StocGradDescent') 
    ax.spines['right'].set_color('none') 
    ax.spines['top'].set_color('none')
    
    
    #############################图3##################################
    ax = fig.add_subplot(223) 
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s') 
    ax.scatter(xcord2,ycord2,s=30,c='blue',) 
    x = arange(-3.0,3.0,0.1) 
    y_3 = (-weights_3[0]-weights_3[1]*x)/(weights_3[2]) 
    ax.plot(x,y_3,'k--',color = 'yellow',linewidth=2) 
    plt.xlabel('Logistics Regression StocGradDescent')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    
    
    
    plt.show()
    
# 定义准确度计算函数 
def calAccuracyRate(dataMat,labelMat,weights): 
    count = 0 
    dataMat = mat(dataMat) 
    labelMat = mat(labelMat).T 
    m,n = shape(dataMat) 
    for i in range(m): 
        h = sigmoid(dataMat[i,:] * weights) 
        if ( h>0.5 and int(labelMat[i,0]) == 1) or ( h<0.5 and int(labelMat[i,0]) == 0 ): 
            count += 1 
    return count/m

dataMat,labelMat = loadDataSet() 
weights_GD = gardDescent(dataMat,labelMat) # 使用梯度下降计算参数矩阵 θ 
weights_SGD = stocGradAscent0(dataMat,labelMat) # 使用随机梯度下降计算参数矩阵 θ 
weights_SGD1 = stocGradAscent1(dataMat,labelMat)
print('weights_GD:\n',weights_GD) 
print('weights_SGD:\n',weights_SGD) 
plotBestFit(weights_GD,weights_SGD,weights_SGD1) # 计算两种算法结果的准确度 
acc_gd = calAccuracyRate(dataMat,labelMat,weights_GD) 
weights_SGD = mat(weights_SGD).transpose() 
acc_sgd = calAccuracyRate(dataMat,labelMat,weights_SGD) 
weights_SGD1 = mat(weights_SGD1).transpose()
acc_sgd1 = calAccuracyRate(dataMat,labelMat,weights_SGD1)
print('\n\nacc_gd:',acc_gd) 
print('acc_sgd:',acc_sgd)
print('acc_sgd1',acc_sgd1)

#未完。。待续
#https://blog.csdn.net/sinat_34022298/article/details/76943283
#还要找非线性回归的例子  就是那个可以划出一个圆来分类的那种  
#还要找多元分类的那种，就是用几条线来分类的那种
