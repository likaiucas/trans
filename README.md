# 毕业设计代码



若需要运行代码，下载文件夹并解压，选择进入train.py文件，使用写好的get_model函数导入模型，或直接通过修改train.py中的注释，运行所需模型。所使用的联合发布数据集可通过链接下载，https://tianchi.aliyun.com/competition/entrance/531860/information
将数据以文件夹格式组织
-data
 --imgdata
  ----val
  ----train_val
 --labeldata
  ----val
  ----train_val

<br> 配置环境：pytorch=1.7.1, cuda=10.1，torchvision=0.8.2 
<br>硬件设备：
<br>本地计算机gpu：Titan xp\<br>
<br>服务器设备gpu：四张GTX 2080ti\<br>
## myfun文件夹
自行完成并设计算法，实现了存储数据内容，拼接局部影像的函数功能，如果目标数据为tiff需要转化为jpg或png格式，使用dataset_building.py, makedata2jpg.py, enviread.py进行操作。

## 模型评价代码/trans/judgynet.py
如需评价训练出的模型结果，修改模型，以及测试集文件的路径即可。该文件会调用utils中的metric.py文件。
