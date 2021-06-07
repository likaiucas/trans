# 毕业设计代码
第一次学习README.md文件排版，格式正在努力规划ing......
若需要运行代码，下载文件夹并解压，选择进入train.py文件，使用写好的get_model函数导入模型，或直接通过修改train.py中的注释，运行所需模型。所使用的联合发布数据集可通过链接下载，https://tianchi.aliyun.com/competition/entrance/531860/information
将数据以文件夹格式组织
>data
>>imgdata
>>>val

>>>train_val

>>labeldata
>>>val

>>>train_val

<br> 配置环境：pytorch=1.7.1, cuda=10.1，torchvision=0.8.2 
<br>硬件设备：
<br>本地计算机gpu：Titan xp
<br>服务器设备gpu：四张GTX 2080ti
## myfun文件夹
自行完成并设计算法，实现了存储数据内容，拼接局部影像的函数功能，如果目标数据为tiff需要转化为jpg或png格式，使用dataset_building.py, makedata2jpg.py, enviread.py进行操作。

## 模型评价代码/trans/judgynet.py
如需评价训练出的模型结果，修改模型，以及测试集文件的路径即可。该文件会调用utils中的metric.py文件。

## 单图评价代码/trans/evalnet.py
数据的输入为单张的图像，代码的输出为单张各类型精度

## 标签绘制存储/trans/buildlegend.py
输入数据为标签的个数，由于产生label的RGB值函数的内置规则固定，因此，只需输入标签数目便可产生最终结果

## Bagging模型的最终集成/trans/meanmodel.py
输入为待集成的三种模型以及模型的训练出的参数，输出为一个Bagging模型

##  /trans/recolor.py | /trans/recolor_smallmap.py
将彩色索引修改为RGB值的标签文件，分步实现，涉及到模型的拼接问题，因此可以先处理小图，再处理大图。

## /trans/valcut.py
实现了针对一景图像拆分函数，以及一景图像的合并

## /trans/PredALLImage.py
预测文件夹下所有的图片

--------------------------------------------------------------------------------------------------------
# 未完待续......
