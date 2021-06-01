# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 09:42:11 2020

@author: asus123
"""
import gdal
import numpy as np
from mxnet import image
import mxnet

mask_path = r"F:\qq\1714469097\FileRecv\label.png"
data_path =r"F:\qq\1714469097\FileRecv\SM33.tif"
storage_path = r"F:\qq\1714469097\FileRecv\new method\data"

'''
数据的读取部分
'''
class readGeoTif(object):
    def __init__(self, path, band):
        # band为有效波段的一个list
        #path下存储tif文件
        self.path = path
        self.bandseting = band
        self.data = self.readasnumpy()
        
    def Image(self, path, bandnum): 
        # bandnum为想要读取的波段号
        #读为一个numpy数组  
        dataset = gdal.Open(path)
        band = dataset.GetRasterBand(bandnum)
        nXSize = dataset.RasterXSize #列数
        nYSize = dataset.RasterYSize #行数
        data= band.ReadAsArray(0,0,nXSize,nYSize).astype(np.float)
        return data
    
    def readasnumpy(self):
        data=[]
        for bandnum in self.bandseting:
            data.append(self.Image(self.path, bandnum))
        data=np.array(data)
        data=data.transpose(1, 2, 0)
        return data
    
    def ReturnData(self):
        return self.data
    
'''
OneHot编码
'''
def Onehot(data):
    #data为array
    # data = np.array(data)
    listlabel = []
    x, y, z=data.shape
    # listdata = data.tolist()
    OH = np.zeros((x,y,5))    
    for m in range(x):
        for n in range(y):
            it = data[m][n].tolist()
            if (it not in listlabel) and (it != [0,0,0]):
                listlabel.append(it)

    # listlabel.remove(np.array([0,0,0]))
    for m in range(x):
        for n in range(y):
            # if data[m][n] is 
            if data[m][n].tolist()!=[0,0,0]:
                label=listlabel.index(data[m][n].tolist())
                OH[m][n][label]=1.
    return OH           
    
'''
数据的随机裁剪
'''
def crop(mask_path, data, size, number):
    mask=image.imread(mask_path)
    # mask = Onehot(mask.asnumpy())
    mask=mxnet.nd.array(mask)
    datalist=[]
    labellist=[]
    data = mxnet.nd.array(data)
    for n in range(number):
        D, rect = image.random_crop(data, (size[0], size[1]))
        datalist.append(D.asnumpy())
        labellist.append(image.fixed_crop(mask, *rect).asnumpy())
    return datalist, labellist

'''
数据转存到磁盘
'''
if __name__ == "__main__":
    Geotif=readGeoTif(data_path, [1, 2, 3, 4, 5, 6, 7, 9])
    data = Geotif.ReturnData()
    
    datalist, labellist = crop(mask_path, data, [32, 32], 300)
    datalist2=[data.transpose(2,0,1) for data in datalist]
    labellist2=[data.transpose(2,0,1) for data in labellist]
    np.save(storage_path + r'\data.npy', np.array(datalist2))
    np.save(storage_path + r'\label.npy', np.array(labellist2))

    datalist, labellist = crop(mask_path, data, [32, 32], 100)
    datalist2=[data.transpose(2,0,1) for data in datalist]
    labellist2=[data.transpose(2,0,1) for data in labellist]
    np.save(storage_path + r'\tstdata.npy', np.array(datalist2))
    np.save(storage_path + r'\tstlabel.npy', np.array(labellist2)) 

   
    # # 读取测试
    # a=np.load(storage_path + r'\data.npy')
    # a=a.tolist()
    # a=[np.array(elem) for elem in a]
    #
    '''
    (batchsize, channel, x, y)
    '''