import os
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image

store_path=r"D:\kevin\myfucUsefulTxt.txt"
ann_path2 = r'D:\kevin\valcut\valcut_out\ann_cut'
img_path2 = r'D:\kevin\valcut\valcut_out\img_cut'
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
        
        
        def crop(t, i):
            tt_ans, ii_ans = [], []
            x, y = t.shape
            m, n = 0, 0
            for xx in range(0, x-self.size, self.size):
                for yy in range(0, y-self.size, self.size):
                    tt, ii = t[xx:xx+self.size, yy:yy+self.size], i[xx:xx+self.size, yy:yy+self.size, :]
                    ann=os.path.join(ann_path2, str(m).zfill(4)+"_"+str(n).zfill(4)+".png")
                    img=os.path.join(img_path2, str(m).zfill(4)+"_"+str(n).zfill(4)+".png")
                    ann_tru=Image.fromarray(tt)
                    ann_tru.save(ann)
                    img_vis=Image.fromarray(ii)
                    img_vis.save(img)                    
                    n+=1
                m+=1
                    
            return tt_ans, ii_ans
        
        crop_fun=crop
        crop_fun(target, img)
    
    def combine(self, im_path, store_path=r'D:\kevin\valcut\ans', name='1'):
        flag=False
        def stick(im_list):
            ans=[]
            for imline in im_list:
                w=imline[0].transpose(0,2,1)

                for im in imline[1:]:
                    im=im.transpose(0,2,1)
                    w=np.c_[w,im]
                w=w.transpose(0,2,1)
                ans.append(w)
            
            w=ans[0]
            for imline in ans[1:]:
                w=np.r_[w, imline]
            
            return w
                    
        for root, dirs, files in os.walk(im_path):
            check=files[0].split("_")[0]
            temp=[]
            im_list=[]
            for file in files:
                a, b=file.split("_")
                if check==a:
                    p=os.path.join(root,file)
                    ii = Image.open(p)
                    ii = np.array(ii)  
                    if len(ii.shape)==2:
                        ii=np.array([ii,ii,ii]).transpose(1,2,0)
                        
                        flag=True
                    temp.append(ii)
                else:
                    check=a
                    im_list.append(temp)
                    p=os.path.join(root,file)
                    ii = Image.open(p)
                    ii = np.array(ii)   
                    if len(ii.shape)==2:
                        ii=np.array([ii,ii,ii]).transpose(1,2,0)
                        flag=True
                    temp=[ii]
            img=stick(im_list)
            if flag:
                img=img[:,:,0]
            img_vis=Image.fromarray(img)
            img_vis.save(os.path.join(store_path, name+'.png'))

        
        
if __name__=="__main__":
    GdalImageProcess = GdalImg()
    GdalImageProcess.combine(r"D:\kevin\valcut\valcut_out\ann_cut2")  

    
    # target_path=r"D:\kevin\Naimg_single\Naimg_single\imgs\p012_r024_l8_20170716.img"
    # val_path=r"D:\kevin\Naimg_single\Naimg_single\gts\p012_r024_l8_20170716.tif"
    # GdalImageProcess.update(val_path,target_path, )
    # print("成功建立数据集")
    
    # GdalImageProcess.combine(r'D:\kevin\valcut\valcut_out\img_cut')    
    
    
    
    
    