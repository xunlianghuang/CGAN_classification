from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#创建文件夹
import os
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder...  ---")
    else:
        print("---  There is this folder!  ---")
save_path="data\\train"
for i in range(10):
    file=os.path.join(save_path,str(i)+'augmentation')
    mkdir(file)
import argparse
import json

import model
import numpy as np
import pylib
import PIL.Image as Image
import tensorboardX
import torch
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as tforms
import torchlib
parser = argparse.ArgumentParser()

parser.add_argument('--z_dim', dest='z_dim', type=int, default=100)#噪声维度
parser.add_argument('--c_dim',dest='c_dim',type=int,default=10)#生成10个类
parser.add_argument('--c_class',dest='c_class',type=int,default=0)#要生成那个类
parser.add_argument('--num',dest='num',type=int,default=20)#要生成数量*100,以10为例就是1000
parser.add_argument('--experiment_name', dest='experiment_name', default='CGAN_default')

args = parser.parse_args()

use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")
c_dim = args.c_dim
z_dim = args.z_dim
c_class = args.c_class
experiment_name = args.experiment_name
G = model.GeneratorCGAN(z_dim=z_dim, c_dim=c_dim).to(device)
# load checkpoint
ckpt_dir = './output/%s/checkpoints' % experiment_name
pylib.mkdir(ckpt_dir)
#加载生成模型
try:
    ckpt = torchlib.load_checkpoint(ckpt_dir)
    start_ep = ckpt['epoch']
    G.load_state_dict(ckpt['G'])
except:
    print(' [*] No checkpoint!')
    start_ep = 0
G.eval()#进入测试模式
num = 0
from PIL import Image
for i in  range(10):
    z_sample = torch.randn(c_dim * 10, z_dim).to(device)#[100,100]
    label=np.zeros((c_dim*10,c_dim))
    label[:,c_class]=1
    c_sample = torch.tensor(label, dtype=z_sample.dtype).to(device)#[100,10]
    x_f_sample = (G(z_sample, c_sample) + 1) / 2.0#生成的图片
    for i in range(x_f_sample.shape[0]):
        img = x_f_sample[i]*255.0
        img = img.int()
        img = img.permute(1, 2, 0).cpu().detach().numpy()

        img = Image.fromarray(img.astype(np.uint8))
        img = img.convert('L')
        img = img.resize((28,28))
        img_save_path = os.path.join(save_path, str(c_class)+'augmentation' , str(num)+".jpg")
        img.save(img_save_path)
        num=num+1

