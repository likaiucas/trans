# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 16:30:53 2021

@author: Administrator
"""
import os
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image

store_path=r"D:\kevin\myfucUsefulTxt.txt"

# 数据转换字典
cfg={0:0, 10:1, 20:2, 30:3, 40:4, 50:5, 60:6, 70:0, 80:0, 90:0, 100:7, 255:8}

root_folder = r"D:\kevin\final_dataset"

def getpath(path):
    imgpath=path.replace("tif", "img").replace("gts", "imgs")
    return path, imgpath
    

class GdalImg():
    def __init__(self, num_of_crop=100, random_crop=False, size=256):
        self.num_of_crop = num_of_crop
        self.random_crop = random_crop
        self.size=size
        self.idx=0
        
    def update(self, target_path, img_path):
        # 读取数据部分
        def image(path, bandnum=[3,2,1]): 
            # bandnum为想要读取的波段号
            # 读为一个numpy数组 
            ans=[]
            dataset = gdal.Open(path)
            for bn in bandnum:
                band = dataset.GetRasterBand(bn)
                nXSize = dataset.RasterXSize #列数
                nYSize = dataset.RasterYSize #行数
                data= band.ReadAsArray(0,0,nXSize,nYSize).astype(np.uint8)
                ans.append(data)
            return np.array(ans)
        target = image(target_path, [1])
        img = image(img_path)
        
        target = np.squeeze(target, axis=0)
        img = img.transpose(1,2,0)
        
        # 依据字典转换
        for k in cfg.keys():
            target[target==k]=cfg[k]
        
        # 使用不同的crop方法
        def randomcrop(t,i, n=self.num_of_crop):
            tt_ans, ii_ans = [], []
            x, y = t.shape            
            for nn in range(n):
                seed_x, seed_y = random.randint(0,x-self.size-1), random.randint(0,y-self.size-1)
                tt, ii = t[seed_x : seed_x+self.size, seed_y : seed_y+self.size], i[seed_x : seed_x+self.size, seed_y : seed_y+self.size, :]
                # tt=np.squeeze(tt, axis=0)
                # ii=ii.transpose(1,2,0)
                tt_ans.append(tt)
                ii_ans.append(ii)
            return tt_ans, ii_ans
        
        def crop(t, i):
            tt_ans, ii_ans = [], []
            x, y = t.shape
            for xx in range(0, x-self.size, self.size):
                for yy in range(0, y-self.size, self.size):
                    tt, ii = t[xx:xx+self.size, yy:yy+self.size], i[xx:xx+self.size, yy:yy+self.size, :]
                    # tt=np.squeeze(tt, axis=0)
                    # ii=ii.transpose(1,2,0)                    
                    tt_ans.append(tt)
                    ii_ans.append(ii)                    
            return tt_ans, ii_ans
        
        crop_dic = {True: randomcrop, False: crop}
        crop_fun=crop_dic[self.random_crop]
        
        tt_ans, ii_ans = crop_fun(target, img)
        dataset = list(zip(tt_ans, ii_ans))
        random.shuffle(dataset,)
        
        # 存储分别数据为 训练集 和 测试集
        def store(ann_path, img_path, dataset):
            for (tt, ii) in dataset:
                ann=os.path.join(ann_path, str(self.idx).zfill(8)+".png")
                img=os.path.join(img_path, str(self.idx).zfill(8)+".png")
                
                ann_tru=Image.fromarray(tt)
                ann_tru.save(ann)
                img_vis=Image.fromarray(ii)
                img_vis.save(img)
                
                self.idx+=1
        
        # trainset
        ann_path=r"D:\kevin\showimg\ann_dir\train_val"
        img_path=r"D:\kevin\showimg\img_dir\train_val"
        store(ann_path, img_path, dataset[len(dataset)//4:])
        
        # testset
        ann_path=r"D:\kevin\showimg\ann_dir\val"
        img_path=r"D:\kevin\showimg\img_dir\val"
        store(ann_path, img_path, dataset[0:len(dataset)//4])        
        
        
if __name__=="__main__":
    GdalImageProcess = GdalImg()
    # with open(store_path) as file: 
    #     n=1
    #     while n<30:
    #         line = file.readline()
    #         if not line:
    #             break
    #         print(line) # do something
    #         target, val = getpath(line)
    #         target = target.replace("\n","")
    #         val = val.replace("\n","")
    #         GdalImageProcess.update(target, val)
    #         n+=1
    target_path=r"D:\kevin\Naimg_single\Naimg_single\imgs\p012_r024_l8_20170716.img"
    val_path=r"D:\kevin\Naimg_single\Naimg_single\gts\p012_r024_l8_20170716.tif"
    GdalImageProcess.update(target_path, val_path)
    print("成功建立数据集")
    
    
    
    
    
    
    