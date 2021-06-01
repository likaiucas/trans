# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 12:12:59 2020

@author: asus123
"""

import numpy as np
from osgeo import gdal
import os
from PIL import Image

# 数据转换字典
cfg={0:0, 10:1, 20:2, 30:3, 40:4, 50:5, 60:6, 70:7, 80:8, 90:9, 255:10}

def image(path, bandnum): 
    # bandnum为想要读取的波段号
    #读为一个numpy数组  
    dataset = gdal.Open(path)
    band = dataset.GetRasterBand(bandnum)
    nXSize = dataset.RasterXSize #列数
    nYSize = dataset.RasterYSize #行数
    data= band.ReadAsArray(0,0,nXSize,nYSize).astype(np.int64)
    return data

class statistics():
    def __init__(self, ):
        # self.nowdata=None
        self.Integral_Class_Distribution={n : 0 for n in range(11)}
        self.now_data={n : 0 for n in range(11)}
        self.current_flag=1
        self.path_list=[]
        self.select_area={n : 0 for n in range(11)}
        
        
    def update(self, path=None, matrix=None):
        #每获得一个数据对矩阵进行更新
        if not matrix:
            matrix=readval(path).flatten()
            
        count = np.bincount(matrix,)
        count=count/np.sum(count)
        for i, d in enumerate(count):
            if cfg.get(i):
                index=cfg.get(i)
                self.Integral_Class_Distribution[index]+=d
                self.now_data[index]=d
                self.current_flag*=d
                
        if self.current_flag and path:
            self.path_list.append(path)
        self.current_flag=1
        
        #测试小样本类
        if (self.now_data[8]):
            self.path_list.append(path)
            for k in cfg.values():
                self.select_area[k] += self.now_data[k]
    
    def getdata(self,):
        return self.path_list, self.Integral_Class_Distribution, self.select_area
    
    def getnowdata(self,):
        return self.now_data
        
    def reset(self,):
        self.Integral_Class_Distribution={n : 0 for n in range(11)}
        self.now_data={n : 0 for n in range(11)}
        self.current_flag=1
        self.path_list=[]
        self.select_area={n : 0 for n in range(11)}
def readval(path):
    return image(path,1)


def Normornize(band):
    return 255*(band-np.min(band))/np.max(band-np.min(band))

def writefile(data, store_path=r"D:\kevin\myfucUsefulTxt.txt"):
    seperate_char = "\n"
    f=open(store_path, "w")
    f.write(seperate_char.join(data))
    f.close()
    

if __name__=='__main__':
    s=statistics()
    dirfor_img = r"D:\kevin\Naimg_single\Naimg_single\gts"
    
    for root, dirs, files in os.walk(dirfor_img):
        for name in files[2:]:
            s.update(os.path.join(root, name))
    
    path_list, Integral_Class_Distribution, select_area = s.getdata()
    print("可使用的文件路径：", path_list)
    print("整体数据分布：", Integral_Class_Distribution)
    print("选择区域的分布：", select_area)
    writefile(path_list)

    
    
    
    
    
    
    
    
    