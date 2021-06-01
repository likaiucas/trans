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
#
import segmentation_models_pytorch as smp
Image.MAX_IMAGE_PIXELS = 1000000000000000

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda")

def rename(d):
    return {k.replace("module.",""):v for k,v in d.items()}


# 准备数据集
data_dir = r"D:\kevin\final_dataset5"
train_imgs_dir = os.path.join(data_dir, "img_dir/train_val/")
val_imgs_dir = os.path.join(data_dir, "img_dir/val/")

train_labels_dir = os.path.join(data_dir, "ann_dir/train_val/")
val_labels_dir = os.path.join(data_dir, "ann_dir/val/")

train_data = RSCDataset(train_imgs_dir, train_labels_dir, transform=train_transform)
valid_data = RSCDataset(val_imgs_dir, val_labels_dir, transform=val_transform)

# 网络

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
#网络2 HRNet


""" "Unet++"

"""
model_name = 'efficientnet-b6'#xception
n_class=11
model=seg_qyl(model_name,n_class)
Unetpp=model.to(device)
# model= torch.nn.DataParallel(model)
""" "Unet"


"""
from MyUnet import MyUnet
model=MyUnet.Unet()
model_name="Vgg16Unet-new"
Unet=model.to(device)
""" "HRnet"

# model= torch.nn.DataParallel(model)
"""
from MyHRnet.config import config
from MyHRnet import seg_hrnet
model_name="HRnet"
model = seg_hrnet.get_seg_model(config)
HRNet=model.to(device)
""" "Deeplab V3+系列"


"""
from MyDeeplab.deeplab import *
model=DeepLab(backbone="resnet", output_stride=16, num_classes=9) #backbone: mobilenet, xception, resnet, drn
DeepLab_resnet = model.to(device)

model=DeepLab(backbone="drn", output_stride=16, num_classes=9) #backbone: mobilenet, xception, resnet, drn
DeepLab_drn = model.to(device)

model=DeepLab(backbone="mobilenet", output_stride=16, num_classes=9) #backbone: mobilenet, xception, resnet, drn
DeepLab_mobilenet = model.to(device)

model=DeepLab(backbone="xception", output_stride=16, num_classes=9) #backbone: mobilenet, xception, resnet, drn
DeepLab_xception = model.to(device)
# m1_CKPT=torch.load(r"D:/kevin/data-science-competition-main/天池/2021全国数字生态创新大赛-高分辨率遥感影像分割/outputs/DeeplabV3Plus-resnet/ckpt/resnet.pth", )
# m1_CKPT = rename(m1_CKPT['state_dict'])
# model= torch.nn.DataParallel(model)
# model= torch.nn.DataParallel(model)
# checkpoints=torch.load('outputs/efficientnet-b6-3729/ckpt/checkpoint-epoch20.pth')
# model.load_state_dict(checkpoints['state_dict'])

model_dict = {"Unet": Unet, 
              "HRNet":HRNet,
              "Unetpp":Unetpp,
              "DeepLab_drn":DeepLab_drn,
              "DeepLab_xception":DeepLab_xception,
              "DeepLab_mobilenet":DeepLab_mobilenet,
              "DeepLab_resnet":DeepLab_resnet,
    }

path = {"DeepLab_drn":r"D:/kevin/data-science-competition-main/天池/2021全国数字生态创新大赛-高分辨率遥感影像分割/outputs/DeeplabV3Plus-drn-new/ckpt/checkpoint-best.pth",
        "DeepLab_xception":r'D:/kevin/data-science-competition-main/天池/2021全国数字生态创新大赛-高分辨率遥感影像分割/outputs/DeeplabV3Plus-xception-new/ckpt/checkpoint-best.pth',
        "DeepLab_mobilenet":r'D:/kevin/data-science-competition-main/天池/2021全国数字生态创新大赛-高分辨率遥感影像分割/outputs/DeeplabV3Plus-mobilenet-new/ckpt/checkpoint-best.pth',
        "DeepLab_resnet":r'D:/kevin/data-science-competition-main/天池/2021全国数字生态创新大赛-高分辨率遥感影像分割/outputs/DeeplabV3Plus-resnet-new/ckpt/checkpoint-best.pth',
        "Unetpp":r'D:/kevin/data-science-competition-main/天池/2021全国数字生态创新大赛-高分辨率遥感影像分割/outputs/efficientnet-b6/ckpt/checkpoint-best.pth',
        "HRNet":r'D:/kevin/data-science-competition-main/天池/2021全国数字生态创新大赛-高分辨率遥感影像分割/outputs/HRnet/ckpt/checkpoint-best.pth',
        "Unet":r'D:/kevin/data-science-competition-main/天池/2021全国数字生态创新大赛-高分辨率遥感影像分割/outputs/Vgg16Unet-new/ckpt/checkpoint-best.pth',
        }

def returnmodel(model_name, pretrained=None):
    model=model_dict[model_name]
    if pretrained:
        m1_CKPT=torch.load(pretrained, )
        m1_CKPT = rename(m1_CKPT['state_dict'])
        model.load_state_dict(m1_CKPT)
    else:
        m1_CKPT=torch.load(path[model_name], )
        m1_CKPT = rename(m1_CKPT['state_dict'])
        model.load_state_dict(m1_CKPT)        
    return model

class meanmodel(nn.Module):
    def __init__(self, model_list):
        super().__init__() 
        self.model=model_list
    def forward(self,x):
        out=[]
        for model in self.model:
            out.append(model(x))
        
        ans=out.pop()
        while out:
            ans+=out.pop()[:,0:8,:,:]
        return ans 
            
            
            
            
def return_mean_model(model_name_list):
    model_list=[]
    for name in model_name_list:
        model_list.append(returnmodel(name))
    return meanmodel(model_list)

