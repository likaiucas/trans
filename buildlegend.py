# -*- coding: utf-8 -*-
"""
Created on Mon May 10 18:53:03 2021

@author: Administrator
"""
from PIL import Image
import numpy as np
import os

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

def getlAabel(n_class, path):
    color_map=get_color_map_list(n_class)
    for n in range(n_class):
        color=color_map[(3*n): (3*n+3)]
        img=np.array(color*28*56).reshape(28,56,3)
        img=Image.fromarray(np.uint8(img))
        img.save(os.path.join(path, str(n)+'.png'))

if __name__=="__main__":
    getlAabel(11, path=r"D:\kevin\valcut\legend\legend1")
    getlAabel(9, path=r"D:\kevin\valcut\legend\legend2")    
        
        
        