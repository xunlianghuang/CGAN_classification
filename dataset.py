# coding=utf-8
import random
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
train_transformer = transforms.Compose([
    transforms.ToTensor()])  # transform it into a torch tensor
class SIGNSDataset(Dataset):
    def __init__(self, data_dir='data\\trian', transform=train_transformer,augmentation=True):

        filenames = os.listdir(data_dir)
        self.img = []
        self.labels = []
        for name in filenames:
            if len(name)>1 and augmentation==False:
               continue
            img_path = os.path.join(data_dir,name)
            img_names_list = os.listdir(img_path)
            for n in img_names_list:
                img_path_ = os.path.join(img_path,n)
                self.img.append(img_path_)
            length = len(img_names_list)
            str_label = name[0]
            self.labels = self.labels+([int(str_label)]*length)
        self.transform = transform

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        image = Image.open(self.img[idx])  # PIL image
        image = self.transform(image)
        label = torch.tensor(self.labels[idx]).type(torch.long)
        return image, label
if __name__ == "__main__":
    train_transformer = transforms.Compose([
        transforms.ToTensor()])  # transform it into a torch tensor
    dataset_test = SIGNSDataset(data_dir="data/train",transform=train_transformer,augmentation=False)
    print(dataset_test[9999])
    print(1)