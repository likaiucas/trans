# -*- coding: utf-8 -*-
"""
Created on Sun May  9 20:08:31 2021

@author: Administrator
"""
from PIL import Image

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
    img=Image.open(r"D:\kevin\valcut\ans\1.png")
    cfg={0:0, 10:1, 20:2, 30:3, 40:4, 50:5, 60:6, 70:0, 80:0, 90:0, 100:7, 255:8}
    
    color_map = get_color_map_list(9)
    img.putpalette(color_map)
    # img.show()
    img.save(r'D:\kevin\valcut\ans\4.png')

