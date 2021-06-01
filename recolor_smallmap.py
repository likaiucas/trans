# -*- coding: utf-8 -*-
"""
Created on Mon May 10 11:11:00 2021

@author: Administrator
"""
import os
from PIL import Image

ann_path = r"D:\kevin\valcut\valcut_out\ann_cut2"
prd_path = r"D:\kevin\valcut\valcut_out\predans3"   
r_ann = r"D:\kevin\valcut\valcut_out\recolor_ann3"
r_prd = r"D:\kevin\valcut\valcut_out\recolor_pred3"
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

color_map=get_color_map_list(9)
if  __name__=="__main__":
    for root, dirs, files in os.walk(ann_path):
        for file in files:
            file_path = os.path.join(root, file)
            img = Image.open(file_path)
            img.putpalette(color_map)
            img.save(os.path.join(r_ann, file))
            
    for root, dirs, files in os.walk(prd_path):
        for file in files:
            file_path = os.path.join(root, file)
            img = Image.open(file_path)
            img.putpalette(color_map, )
            img.save(os.path.join(r_prd, file))