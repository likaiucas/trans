# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 11:14:01 2021

@author: Administrator
"""
import numpy as np

def sigmoid(z):
    return 1.0/(1+np.exp(-z))

def dSigmoid(a):
    return np.multiply(a,1-a)

def initThetas(hiddenNum, unitNum, inputSize, classNum, epsilon):
    """

    Parameters
    ----------
    hiddenNum : TYPE
        DESCRIPTION.隐层数目
    unitNum : TYPE
        DESCRIPTION.每个隐层的神经元数目
    inputSize : TYPE
        DESCRIPTION.输入层规模
    classNum : TYPE
        DESCRIPTION.分类数目
    epsilon : TYPE
        DESCRIPTION.随机的邻域大小

    Returns
    -------
    Theta：权值矩阵序列

    """
    
    hiddens = [unitNum for i in range(hiddenNum)]
    units = [inputSize] + hiddens + [classNum] 
    Thetas = []
    for idx, unit in enumerate(units):
        if idx == len(units)-1:
            break
        
        nextUnit=units[idx+1]
        
        Theta = np.random.rand(nextUnit, unit + 1)*2*epsilon - epsilon
        Thetas.append(Theta)
    return Thetas

def computeCost(Thetas, y, theLambda, X=None, a=None):
    """计算代价

    Parameters
    ----------
    Thetas : TYPE
        DESCRIPTION.权值矩阵序列
    y : TYPE
        DESCRIPTION.标签集
    theLambda : TYPE
        DESCRIPTION.正则化项
    X : TYPE, optional
        DESCRIPTION. 样本，The default is None.
    a : TYPE, optional
        DESCRIPTION. 各层激活值，The default is None.

    Returns
    -------
    J 预测代价

    """
    m=y.shape[0]
    if a is None:
        a = np.fp(Thetas, X)
        
    error = -np.sum(np.multiply(y.T, np.log(a[-1])) + np.multiply((1-y).T, np.log(1-a[-1])))
    
    reg = -np.sum([np.sum(Theta[:, 1:]) for Theta in Thetas])
    return (1.0/m)*error + (1.0/(2*m))*theLambda * reg

def adjustLabels(y):
    """
    Parameters
    ----------
    y : TYPE
        DESCRIPTION.标签集

    Returns
    -------
    yAdjusted 向量化后的标签

    """
    if y.shape[1]==1:
        classes = set(np.ravel(y))
        classNum = len(classes)
        minClass = min(classes)
        if classNum>2:#多分类
            yadjusted = np.zeros((y.shape[0], classNum), np.float64)
            for row, label in enumerate(y):
                yadjusted[row,label-minClass] = 1
        else:
            yadjusted = np.zeros((y.shape[0], 1), np.float64)
            for row, label in enumerate(y):
                if label !=minClass:
                    yadjusted[row,0] = 1.0
            return yadjusted
    return y

def unroll(matrixes):
    """
    Parameters
    ----------
    matrixes : TYPE
        DESCRIPTION.矩阵

    Returns
    -------
    vec 向量

    """
    vec=[]
    for matrix in matrixes:
        vector = matrix.reshape(1, -1)[0]
        vec = np.concatenate((vec, vector))
    return vec

def roll(vector, shapes):
    matrixes=[]
    begin = 0
    for shape in shapes:
        end = begin +shape[0]*shape[1]
        matrix = vector[begin:end].reshape[shape]
        begin = end
        matrixes.append(matrix)
    return matrixes


def fp(Thetas, X):
    """前向传播
    Parameters
    ----------
    Thetas : TYPE
        DESCRIPTION.矩阵权值
    x : TYPE
        DESCRIPTION.输入样本
    Returns
    -------
    a输出值
    """
    
    layers=range(len(Thetas)+1)
    layerNum=len(layers)
    
    #激活向量序列
    a=range(layerNum)
    
    for l in layers:
        if l==0:
            a[l]=X.T
        else:
            z = Thetas[l-1]*a[l-1]
            a[l] = sigmoid(z)
        
        if l !=layerNum-1:
            a[l]=np.concatenate((np.ones((1, a[1].shape[1])), a[l]))
        return a
    
def bp(Thetas, a, y, theLambda):
    """反向传播
        Parameters
    ----------
    Thetas : TYPE
        DESCRIPTION.
    a : TYPE
        DESCRIPTION.激活值
    y : TYPE
        DESCRIPTION.
    theLambda : TYPE
        DESCRIPTION.

    Returns
    -------
    D 权值梯度.

    """
    m = y.shape[0]
    layers = range(len(Thetas) +1)
    layerNum = len(layers)
    d = range(len(layers))
    delta = [np.zeros(Theta.shape) for Theta in Thetas]
    
    for l in layers[::-1]:
        if l==0:
            break
        if l==layerNum - 1:
            d[l] = a[l] - y.T
        else:
            d[l] = np.multiply((Thetas[l][:,1:].T*d[l+1]), dSigmoid(a[l][1:, :]))
    
    for l in layers[0:layerNum - 1]:
        delta[l]=d[l+1] * (a[l].T)
        
    D = [np.zeros(Theta.shape) for Theta in Thetas]
    for l in range(len(Thetas)):
        Theta = Thetas[l]
        #偏置更新增量
        D[l][:, 0] = (1.0/m)*((delta[l][0:, 0]).reshape(1,-1))
        #权值更新增量
        D[l][:, 1:] = (1.0/m)*(delta[l][0:,1:]+theLambda*Theta[:, 1:])
    return D
        
def updateThetas(m, Thetas, D, alpha, theLambda):
    """
    Parameters
    ----------
    m : TYPE
        DESCRIPTION.样本数
    Thetas : TYPE
        DESCRIPTION.各层权值矩阵
    D : TYPE
        DESCRIPTION.梯度
    alpha : TYPE
        DESCRIPTION.学习率
    theLambda : TYPE
        DESCRIPTION.正则化参数

    Returns
    -------
    Thetas. 更新后的权值矩阵

    """
    for l in range(len(Thetas)):
        Thetas[l] = Thetas[l] - alpha*D[l]
    return Thetas

def gradientDescent(Thetas, X, y, alpha, theLambda):
    m,n = X.shape
    a = fp(Thetas, X)
    D = bp(Thetas, a, y, theLambda)
    J = computeCost(Thetas, y, theLambda, a=a)
    Thetas = updateThetas(m, Thetas, D, alpha, theLambda)
    if np.isnan(J):
        J=np.inf
    return J, Thetas

def gradientCheck(Thetas, X,y,theLambda):
    #梯度校验
    m,n =X.shape
    a = fp(Thetas, X)
    
    D=bp(Thetas, a, y, theLambda)
    
    J = computeCost(Thetas, y, theLambda, a=a)
    DVec = unroll(D)
    
    epsilon = 1e-4
    gradApprox=np.zeros(DVec.shape)
    ThetaVec = unroll(Thetas)
    shapes = [Theta.shape for Theta in Thetas]
    for i, item in enumerate(ThetaVec):
        ThetaVec[i] = item -epsilon
        JMinus = computeCost(roll(ThetaVec, shapes), y, theLambda, X=X)
        ThetaVec[i] = item + epsilon
        JPlus = computeCost(roll(ThetaVec, shapes), y, theLambda, X=X)
        gradApprox[i] = (JPlus-JMinus)/(2*epsilon)
        
    diff = np.average(gradApprox - DVec)
    print("gradient check diff:", diff)
    if diff<1e-5:
        return True
    else:
        return False
    
def train(X, y, checkFlag=False, Thetas=None, hiddenNum=0, unitNum=5, epsilon=1, alpha=1, theLambda=0, precision=0.0001, maxIters=50):
    """

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    checkFlag : TYPE, optional
        DESCRIPTION. The default is False.
    Thetas : TYPE, optional
        DESCRIPTION. The default is None.预训练参数，None时为随机
    hiddenNum : TYPE, optional
        DESCRIPTION. The default is 0.
    unitNum : TYPE, optional
        DESCRIPTION. The default is 5.
    epsilon : TYPE, optional
        DESCRIPTION. The default is 1.
    alpha : TYPE, optional
        DESCRIPTION. The default is 1.
    theLambda : TYPE, optional
        DESCRIPTION. The default is 0.
    precision : TYPE, optional
        DESCRIPTION. The default is 0.0001.
    maxIters : TYPE, optional
        DESCRIPTION. The default is 50.

    Returns
    -------
    None.

    """
    m,n=X.shape
    y=adjustLabels(y)
    classNum=y.shape[1]
    
    if Thetas is None:
        Thetas = initThetas(
            inputSize=n,
            hiddenNum=hiddenNum,
            unitNum=unitNum,
            classNum=classNum,
            epsilon=epsilon,
            )
    
    #梯度检验
    print("梯度检验中......")
    if checkFlag:
        checked = gradientCheck(Thetas, X, y, theLambda)
    else:
        checked=True
    print("已完成梯度检验")
    
    if checked:
        last_error = np.inf
        for i in range(maxIters):
            error, Thetas = gradientDescent(
                Thetas, X, y, alpha=alpha, theLambda=theLambda)
            if abs(error-last_error)<precision:
                last_error = error
                break
            
            if error==np.inf:
                last_error = error
                break
            last_error = error
        return {"error":error,
                "Thetas":Thetas,
                "Iters":i}
    else:
        print("Error:梯度检验出错！！！")
        return {"error":None,
                "Thetas":None,
                "Iters":i}
    
def predict(X, Thetas):
    a=fp(Thetas, X)
    return a[-1]

import matplotlib.pyplot as pyplot
def check_manufacture(idx, X, Thetas,y):
    print("Predict:", (np.argmax(predict(X[idx], Thetas))+1))
    print("Real tag:", y[idx].ravel())
    pyplot.imshow(X[idx].reshape(20, 20).T)