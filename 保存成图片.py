from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torchvision.transforms as transformers
from PIL import Image
import torchvision
import os
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder...  ---")
    else:
        print("---  There is this folder!  ---")
save_path_train="data\\train"
save_path_test="data\\test"
for i in range(10):
    file=os.path.join(save_path_train,str(i))
    mkdir(file)
for i in range(10):
    file=os.path.join(save_path_test,str(i))
    mkdir(file)
fashion_train = torchvision.datasets.FashionMNIST(root=r'data/FashionMNIST',train=True,download=True)
fashion_test = torchvision.datasets.FashionMNIST(root=r'data/FashionMNIST',train=False,download=True)
#生成test的图片
test_length = len(fashion_test)
list = [0,0,0,0,0,0,0,0,0,0]
for i in range(test_length):
    feature, label = fashion_test[i]
    #拼凑路径
    img_save_path=os.path.join(save_path_test,str(label),str(list[label])+'.jpg')
    list[label]=list[label]+1
    feature.save(img_save_path)
print(1)
#生成train的图片
train_length=len(fashion_train)
list = [0,0,0,0,0,0,0,0,0,0]
for i in range(train_length):
    feature, label = fashion_train[i]
    #拼凑路径
    img_save_path=os.path.join(save_path_train,str(label),str(list[label])+'.jpg')
    list[label]=list[label]+1
    feature.save(img_save_path)
print(1)