# coding=utf-8
from mpl_toolkits.axes_grid1 import host_subplot
from datetime import datetime
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt


def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct / total


def train(net, train_data, valid_data, num_epochs, optimizer, criterion):
    model_cp = './model/'  # 网络参数保存位置
    train_val_num = []
    train_acc_plot = []
    val_acc_plot = []
    if torch.cuda.is_available():
        net = net.cuda()
    prev_time = datetime.now()
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        net = net.train()#设置成训练模式
        for im, label in train_data:
            if torch.cuda.is_available():
                im = im.cuda()  # (bs, 3, h, w)
                label = label.cuda()  # (bs, h, w)
            else:
                im = im
                label = label
            # forward
            output = net(im)
            loss = criterion(output, label)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += get_acc(output, label)

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0
            net = net.eval()#设置成评估模式
            for im, label in valid_data:
                with torch.no_grad():
                    if torch.cuda.is_available():
                        im =im.cuda()
                        label = label.cuda()
                    else:
                        im = im
                        label = label
                output = net(im)
                loss = criterion(output, label)
                valid_loss += loss.item()
                valid_acc += get_acc(output, label)
            epoch_str = (
                    "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, "
                    % (epoch, train_loss / len(train_data),
                       train_acc / len(train_data), valid_loss / len(valid_data),
                       valid_acc / len(valid_data)))
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch, train_loss / len(train_data),
                          train_acc / len(train_data)))
        prev_time = cur_time
        train_val_num.append(epoch)
        train_acc_plot.append(float('%.4f' % (train_acc / len(train_data))))
        val_acc_plot.append(float('%.4f' % (valid_acc / len(valid_data))))
        print(epoch_str + time_str)
        if (epoch + 1) % 11 == 0:
            torch.save(net, '{0}/model_{1}.pth'.format(model_cp, epoch))  # 训练所有数据后，保存网络的参数
            print("save {0}/model_{1}.pth successfully!".format(model_cp, epoch))
    host = host_subplot(111)
    plt.subplots_adjust(right=0.8)
    par1 = host.twinx()
    # set labels
    host.set_xlabel("train_val_num")
    host.set_ylabel("train_acc")
    par1.set_ylabel("val_acc")

    # plot curves
    p1, = host.plot(train_val_num, train_acc_plot, label="train_acc")
    p2, = par1.plot(train_val_num, val_acc_plot, label="val_acc")
    host.legend(loc=1)
    # set label color
    host.axis["left"].label.set_color(p1.get_color())
    par1.axis["right"].label.set_color(p2.get_color())
    plt.title('Train_acc & val_acc')
    plt.draw()
    plt.savefig('./train_acc_val_acc.png')
    plt.show()

