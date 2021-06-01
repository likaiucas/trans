# -*- coding: utf-8 -*-
"""
Created on Sat May  2 08:45:33 2020

@author: xsw
"""

import torch
import numpy as np

#对output每个位置求最大索引
def MaxIndex(output):
    #output=torch.rand(64,64,6)
    x, y, z=output.shape
    #print(x,y,z)
    MI = np.zeros((x,y)) 
    for m in range(x): 
            for n in range(y):
                c=0
                i=0
                for k in range(z):
                    if c<output[m][n][k]:
                        c=output[m][n][k]
                        i=k
                MI[m][n]=i
                #print(MI[m][n])
    return MI


#构建混淆矩阵
def ConMat(MI,label):
    CM = np.zeros((6,6)) #共有六类，所以混淆矩阵大小为6*6，其中每一行之和表示该类别的真实样本数量，
    #每一列之和表示被预测为该类别的样本数量。
    x, y=MI.shape
    for m in range(x): 
        for n in range(y):
            for k in range (6):
                for i in range (6):
                    if label[x][y] == k:
                        if MI[x][y]==i:
                            CM[k][i]+=1
    return CM


#求Kappa系数    
def Kappa(CM):
    x, y=CM.shape
    s=CM.sum()
    a=0
    for m in range(x):    #求对角线元素之和
        a+=CM[m][m]   
    b = np.zeros((1,6))
    for m in range(x):    #求每一行元素之和
        for n in range(y): 
            b[0][m]+=CM[m][n]
    c = np.zeros((1,6))
    for n in range(x):    #求每一列元素之和
        for m in range(y): 
            c[0][n]+=CM[m][n]
    d=0
    for k in range(x):
        d+=b[0][k]*c[0][k]
    po=a/s
    pe=d/s**2
    k=(po-pe)/(1-pe)
    return k
   
                    
    
    
    
    
    