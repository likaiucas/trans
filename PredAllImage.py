# -*- coding: utf-8 -*-
"""
Created on Sun May  9 11:07:42 2021

@author: Administrator
"""
import get_model
import os
import torch
from PIL import Image
import numpy as np
import valcut

model=get_model.returnmodel("HRNet",r'outputs/HRnet/ckpt/checkpoint-best.pth')
path=r"D:\kevin\valcut\valcut_out\img_cut2"
store_path = r'D:\kevin\valcut\valcut_out\predans3'
combine_store_path = r'D:\kevin\valcut\ans'
ImageProcess = valcut.GdalImg()

def get_color_map_list(num_classes):
    """ Returns the color map for visualizing the segmentation mask,
        which can support arbitrary number of classes.
    Args:
        num_classes: Number of classes
    Returns:
        The color map
    """
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3

    return color_map



if __name__=="__main__":
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            im = Image.open(file_path)
            im = np.array(im).transpose(2,0,1)
            im=torch.from_numpy(im).unsqueeze(0).cuda().float()
            ans=model(im)
            ans=ans.cpu().squeeze(0).detach().numpy().transpose(1,2,0)
            ans=np.argmax(ans,axis=2)
            img=Image.fromarray(np.uint8(ans))
            img.save(os.path.join(store_path, file.replace('jpg', 'png')))
            # print('suceess!')
    # ImageProcess.combine(store_path, name='predcombine')