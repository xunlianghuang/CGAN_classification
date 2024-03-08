# <p align="center"> Conditional GANs </p>

Pytorch implementation of several GANs with conditional signals (supervised or unsupervised). All experiments are conducted on [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist), and the network structures are adapted from [Improved GAN](https://arxiv.org/abs/1606.03498).

## Conditional GANs

- Supervised
    - [x] [CGAN](http://arxiv.org/abs/1411.1784)
    - [x] [CGAN with Projection Discriminator](http://arxiv.org/abs/1802.05637)
    - [x] [ACGAN](http://arxiv.org/abs/1610.09585)
    - [ ] Others
- Unsupervised
    - [x] [InfoGAN](http://arxiv.org/abs/1606.03657)
    - [ ] Others

## Exemplar Results

CGAN                            | Projection CGAN                 | ACGAN
:---:                           | :---:                           | :---:
<img src="./pics/cgan.jpg">     | <img src="./pics/pcgan.jpg">    | <img src="./pics/acgan.jpg">
**InfoGAN1**                    | **InfoGAN2**                    | **InfoGAN3**
<img src="./pics/infogan1.jpg"> | <img src="./pics/infogan2.jpg"> | <img src="./pics/infogan3.jpg">

## Usage

- Prerequisites
    - PyTorch 1.0.0
    - Python 3.6

- Examples of training
    - training

        ```console
        CUDA_VISIBLE_DEVICES=0 python train_CGAN.py
        ```

    - tensorboard for loss visualization

        ```console
        CUDA_VISIBLE_DEVICES='' tensorboard --logdir ./output/CGAN_default/summaries --port 6006
        ```

- Others
    - If you want to use other datasets, just replace `FashionMNIST` by `MNIST` or `CIFAR10` in the codes.
    - There are arguments for configurations of GAN loss, gradient penalty, and etc, just try them.

-----------------------------------------------------------------

之后运行保存成图片文件

然后使用模型进行数据扩增

之后选择是否扩增进行分类
git init 
git add * #这个是要上传全部的文件 
修改name 和 email
git commit -m "信息"
git remote add origin https://github.com/xunlianghuang/CGAN_classification.git
git push -u origin master
git config --global http.proxy http://127.0.0.1:7890
git config --global https.proxy http://127.0.0.1:7890
git push -u origin master
