# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 15:31:17 2021

@author: Administrator
"""
import os
import torch
import numpy as np
from dataset import RSCDataset
from torch.utils.data import Dataset, DataLoader
from utils import metric
from PIL import Image
from torch.autograd import Variable
from dataset import val_transform
import torch.nn as nn
from torch.cuda.amp import autocast
import segmentation_models_pytorch as smp

import meanmodel

folder=r"D:\kevin\final_dataset5"
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

def loadmodel(model, path=None, backbone=None):
    """
    Parameters
    ----------
    path : str
        DESCRIPTION.
    model : int
        0: HRnet.
        1: Unet
        2: Unet++
        3: DeeplabV3+
    Returns
    -------
    Model.
    """
    if model==0:
        from MyHRnet.config import config
        from MyHRnet import seg_hrnet
        model=seg_hrnet.get_seg_model(config)
    elif model==1:
        from MyUnet import MyUnet
        model=MyUnet.Unet()
    elif model==2:
        from MyUnetPlusPlus import MyUnetPlusPlus
        model=MyUnetPlusPlus.load_unetP(encoder_weights=None)
    elif model==3:
        from MyDeeplab.deeplab import DeepLab
        model=DeepLab(backbone=backbone, output_stride=16, num_classes=11)
    
    if path:
        ckpt=torch.load(path)
        ckpt=rename(ckpt["state_dict"])
        model.load_state_dict(ckpt)
    return model

def get_dataset(folder=folder):
    train_imgs_dir = os.path.join(folder, "img_dir/val/")
    train_labels_dir = os.path.join(folder, "ann_dir/val/")
    data = RSCDataset(train_imgs_dir, train_labels_dir, transform=val_transform)
    loader = DataLoader(dataset=data, batch_size=5, shuffle=False, num_workers=0,drop_last=True)
    return loader

def TransFormat(t,p):
    p=p.cpu().data.numpy()
    p= np.argmax(p,axis=1)
    t=t.cpu().data.numpy()
    return t, p
    
    
def evaluate():
    # model=loadmodel(3,path='D:/kevin/data-science-competition-main/天池/2021全国数字生态创新大赛-高分辨率遥感影像分割/outputs/xception-dpv3.pth',
    #                 backbone="xception")
    
    # model=seg_qyl('efficientnet-b6',9)
    # m1_CKPT=torch.load('D:/kevin/data-science-competition-main/天池/2021全国数字生态创新大赛-高分辨率遥感影像分割/outputs/efficientnet-b6/ckpt/checkpoint-best.pth' )
    # m1_CKPT = rename(m1_CKPT['state_dict'])
    # model.load_state_dict(m1_CKPT)
    
    # from MyDeeplab.deeplab import DeepLab
    # model=DeepLab(backbone='xception', output_stride=16, num_classes=9)
    # m1_CKPT=torch.load(r'outputs/DeeplabV3Plus-xception-new/ckpt/checkpoint-best.pth')
    # m1_CKPT = rename(m1_CKPT['state_dict'])
    
    # from MyUnet import MyUnet
    # model=MyUnet.Unet()    
    # m1_CKPT=torch.load(r'outputs/Vgg16Unet-new/ckpt/checkpoint-best.pth')
    # m1_CKPT = rename(m1_CKPT['state_dict'])
    
    # from MyHRnet.config import config
    # from MyHRnet import seg_hrnet
    # model=seg_hrnet.get_seg_model(config)    
    # m1_CKPT=torch.load(r'outputs/HRnet/ckpt/checkpoint-best.pth')
    # m1_CKPT = rename(m1_CKPT['state_dict'])    
    # model.load_state_dict(m1_CKPT)    
    # model=model.cuda()
    
    model = meanmodel.synmodel()
    
    loader=get_dataset()
    segM=metric.SegMetric(9)
    
    device=torch.device("cuda")
    with torch.no_grad():
        for batch_idx, batch_sample in enumerate(loader):
            data, target = batch_sample['image'], batch_sample['label']
            data, target = Variable(data.to(device)), Variable(target.to(device))
            pred = model(data)
            target, pred = TransFormat(target, pred)
            segM.update(target, pred)
    return segM.get_scores()

if __name__=="__main__":
    import seaborn as sns
    scores=evaluate()
    mtrx=scores['confusion_matrix']
    
    ax = sns.heatmap(mtrx[1:,1:])
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            