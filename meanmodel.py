# -*- coding: utf-8 -*-
"""
Created on Wed May 19 10:46:18 2021

@author: Administrator
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' 

import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from utils import train_net
from dataset import RSCDataset
from dataset import train_transform, val_transform
from torch.cuda.amp import autocast

import segmentation_models_pytorch as smp

def rename(d):
    return {k.replace("module.",""):v for k,v in d.items()}

class seg_qyl(nn.Module):
    def __init__(self, model_name, n_class):
        super().__init__()  
        self.model = smp.UnetPlusPlus(# UnetPlusPlus 
                encoder_name=model_name,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                
                encoder_weights=None,     # use `imagenet` pretrained weights for encoder initialization
                in_channels=3,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
                classes=n_class,                      # model output channels (number of classes in your dataset)
            )
    @autocast()
    def forward(self, x):
        #with autocast():
        x = self.model(x)
        return x

model_name = 'efficientnet-b6'#xception
n_class=9
unetpp=seg_qyl(model_name,n_class)
unetpp=unetpp.cuda()
m1_CKPT=torch.load(r'outputs/efficientnet-b6/ckpt/checkpoint-best.pth', )
m1_CKPT = rename(m1_CKPT['state_dict'])
unetpp.load_state_dict(m1_CKPT)



from MyHRnet.config import config
from MyHRnet import seg_hrnet
import get_model
model_name="HRnet"
HRnet = get_model.returnmodel("HRNet",r'outputs/HRnet/ckpt/checkpoint-best.pth')
# HRnet=HRnet.cuda()
m2_CKPT=torch.load(r'outputs/HRnet/ckpt/checkpoint-best.pth', )
m2_CKPT = rename(m2_CKPT['state_dict'])
HRnet.load_state_dict(m2_CKPT)


from MyDeeplab.deeplab import *
DPResnet=DeepLab(backbone="resnet", output_stride=16, num_classes=9) #backbone: mobilenet, xception, resnet, drn
DPResnet=DPResnet.cuda()

m1_CKPT=torch.load(r"outputs/DeeplabV3Plus-resnet-new/ckpt/checkpoint-best.pth", )
m1_CKPT = rename(m1_CKPT['state_dict'])
DPResnet.load_state_dict(m1_CKPT)


class synmodel(nn.Module):
    def __init__(self,):
        super().__init__()  
        self.HRnet=HRnet
        self.DP=DPResnet
        self.Unetpp=unetpp
    def forward(self,x):
        x1=self.HRnet(x)[:,0:9,:,:]
        # x1=nn.functional.softmax(x1[:,0:8,:,:],dim=1)
        x2=self.DP(x)
        # x2=nn.functional.softmax(x2[:,0:8,:,:],dim=1)
        x3=self.Unetpp(x)
        # x3=nn.functional.softmax(x3[:,0:8,:,:],dim=1)
        x=x1+x2+x3
        return x

if __name__=="__main__":
    model=synmodel()
    out=model(torch.zeros(5,3,256,256).cuda())