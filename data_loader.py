import os
import sys
import numpy as np
from PIL import Image
import json
from tqdm import tqdm
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import torchvision
import torchvision.transforms as transforms


class ImageNetDataset(Dataset):
    def __init__(self,
                 args,
                 image_dir='D:/Academic/tiny-imagenet-200/',
                 Pic2index_file='D:/Academic/tiny-imagenet-200/val_annotations.csv',  # 图片名称→类别index
                 Index2label_file='D:/Academic/tiny-imagenet-200/ImageNetIndex2Label.json',  # 类别index→英文label
                 split='val'):
        super(ImageNetDataset).__init__()
        self.args = args
        self.split = split
        self.image_dir = image_dir + split + ('/' if len(split) else '')
        self.transform = transforms.Compose([
            transforms.Resize(size=(232,232)),
            transforms.CenterCrop(size=(224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.image_list = []

        with open(Pic2index_file, 'r') as file:
            reader = csv.reader(file)
            data = {}
            for row in reader:
                data[row[0]] = row[1:]
            self.Pic2index = data

        with open(Index2label_file, 'r') as f:
            self.Index2label = json.load(f)

        # self.cat_list = sorted(os.listdir(self.image_dir))[:args.num_cat]
        self.cat_list = sorted(os.listdir(self.image_dir))

        for cat in self.cat_list:
            # cat, 类别文件夹list
            # name_list = sorted(os.listdir(self.image_dir + cat))[:self.args.num_perclass]
            name_list = sorted(os.listdir(self.image_dir + cat + ('/images/' if split == 'train' else '')))
            # name_list, 图片名称list
            self.image_list += [self.image_dir + cat + '/' + image_name for image_name in name_list]

        # print('Total %d Data.' % len(self.image_list))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_path = self.image_list[index]
        pic = image_path.split('/')[-1]  # label name
        index = self.Pic2index[pic][0]
        # label=self.Index2label[index]
        # index = torch.LongTensor([index]).squeeze()

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        # prefix = image_path.split('/')[-1].split('.')[0]
        return image, index


class CIFAR10Dataset(Dataset):
    def __init__(self,
                 args,
                 image_dir='/data/user/data/cifar-10-png/',
                 # image_dir='/export/d2/user/dataset/cifar-10-png/',
                 split='train'):
        super(CIFAR10Dataset).__init__()
        self.args = args
        # self.image_dir = image_dir + ('train/' if split == 'train' else 'test/')
        self.image_dir = image_dir + split + ('/' if len(split) else '')
        self.transform = transforms.Compose([
            transforms.Resize(self.args.image_size),
            transforms.CenterCrop(self.args.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ])
        # self.norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
        self.image_list = []
        self.cat_list = sorted(os.listdir(self.image_dir))[:self.args.num_cat]
        for cat in self.cat_list:
            name_list = sorted(os.listdir(self.image_dir + cat))[:self.args.num_perclass]
            self.image_list += [self.image_dir + cat + '/' + image_name for image_name in name_list]

        print('Total %d Data.' % len(self.image_list))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_path = self.image_list[index]
        label = image_path.split('/')[-2]  # label name
        label = self.cat_list.index(label)
        label = torch.LongTensor([label]).squeeze()

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        prefix = image_path.split('/')[-1].split('.')[0]
        return image, label, prefix


class MNISTDataset(Dataset):
    def __init__(self,
                 args,
                 image_dir='/data/user/collection/MNIST/data/mnist_png/',
                 # image_dir='/export/d2/user/dataset/MNIST/mnist_png/',
                 split='train'):
        super(MNISTDataset).__init__()
        self.args = args
        # self.image_dir = image_dir + ('train/' if split == 'train' else 'test/')
        self.image_dir = image_dir + split + ('/' if len(split) else '')
        self.transform = transforms.Compose([
            transforms.Resize(self.args.image_size),
            transforms.CenterCrop(self.args.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        # self.norm = transforms.Normalize((0.1307, ), (0.3081, ))
        self.image_list = []
        self.cat_list = sorted(os.listdir(self.image_dir))[:self.args.num_cat]
        for cat in self.cat_list:
            name_list = sorted(os.listdir(self.image_dir + cat))[:self.args.num_perclass]
            self.image_list += [self.image_dir + cat + '/' + image_name for image_name in name_list]

        print('Total %d Data.' % len(self.image_list))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_path = self.image_list[index]
        label = image_path.split('/')[-2]  # label name
        label = self.cat_list.index(label)
        label = torch.LongTensor([label]).squeeze()

        image = Image.open(image_path)  # .convert('RGB')
        image = self.transform(image)

        prefix = image_path.split('/')[-1].split('.')[0]
        return image, label, prefix


class DataLoader(object):
    def __init__(self, args):
        self.args = args

    def get_loader(self, dataset, shuffle=True):
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.args.batch_size,
            num_workers=int(self.args.num_workers), shuffle=shuffle
        )
        return data_loader


if __name__ == '__main__':
    pass
