# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 19:53:56 2021

@author: Administrator
"""
import os
from shutil import copy
from PIL import Image

All_img_path="D:/kevin/suichang_round1_train_210120/"
name_list=os.listdir(All_img_path)

LenOfData = len(name_list)//2

for n in range(int(LenOfData*0.8)):
    imname=name_list.pop()
    img=os.path.join(All_img_path, imname)
    label=os.path.join(All_img_path, name_list.pop())    
    
    im=Image.open(img)
    r, g, b, a = im.split()
    im=Image.merge("RGB", (r,g,b))
    newname=imname.split(".")[0]+".jpg"
    im.save(os.path.join(r"D:\kevin\data-science-competition-main\天池\2021全国数字生态创新大赛-高分辨率遥感影像分割\dataset\satellite_jpg\img_dir\train_val", newname))
    
    copy(label, r"D:\kevin\data-science-competition-main\天池\2021全国数字生态创新大赛-高分辨率遥感影像分割\dataset\satellite_jpg\ann_dir\train_val")
    
for n in range(int(LenOfData*0.8), LenOfData):
    imname=name_list.pop()
    img=os.path.join(All_img_path, imname)
    label=os.path.join(All_img_path, name_list.pop())    
    
    im=Image.open(img)
    r, g, b, a = im.split()
    im=Image.merge("RGB", (r,g,b))
    newname=imname.split(".")[0]+".jpg"
    im.save(os.path.join(r"D:\kevin\data-science-competition-main\天池\2021全国数字生态创新大赛-高分辨率遥感影像分割\dataset\satellite_jpg\img_dir\val", newname))
    
    copy(label, r"D:\kevin\data-science-competition-main\天池\2021全国数字生态创新大赛-高分辨率遥感影像分割\dataset\satellite_jpg\ann_dir\val")
    
    