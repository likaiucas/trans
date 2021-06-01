# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 10:46:19 2021

@author: Administrator
"""

import torch
import numpy as np
from MyUnet import MyUnet
from utils import metric
from PIL import Image

impath="D:/kevin/data-science-competition-main/天池/2021全国数字生态创新大赛-高分辨率遥感影像分割/000001.jpg"
tpath="D:/kevin/data-science-competition-main/天池/2021全国数字生态创新大赛-高分辨率遥感影像分割/000001.png"

def rename(d):
    return {k.replace("module.",""):v for k,v in d.items()}

def readim(pathim, patht):
    image=Image.open(pathim,"r")
    mask=Image.open(patht,"r")
    im, ma = np.array(image), np.array(mask)
    im=im.transpose(2,0,1)
    return im, ma

unet=MyUnet.Unet()
ckpt=torch.load("D:/kevin/data-science-competition-main/天池/2021全国数字生态创新大赛-高分辨率遥感影像分割/outputs/Vgg16Unet/ckpt/checkpoint-best.pth")
ckpt=rename(ckpt["state_dict"])
unet.load_state_dict(ckpt)

im,mask=readim(impath, tpath)
im=torch.tensor(im)
im=im.unsqueeze(0)
pred=unet(im.float())

pred=pred.cpu().data.numpy().argmax(axis=1).squeeze(0)
segM=metric.SegMetric(11)
segM.update(mask,pred)
print(segM.get_scores())