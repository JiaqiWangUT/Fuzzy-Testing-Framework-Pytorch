# general
import copy
import itertools
import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet152, ResNet152_Weights

# local files
from captum.attr import DeepLift
import cifar10_models
import imagenet_models
import mnist_models
from data_loader import *
from utils import *
from NeuronCoverage import *

# mutate
from Mutation_Strategy import mutation
import imgaug.augmenters as iaa
import torchattacks
from style_operator import Stylized
import image_transforms


def random_pick(arr):
    idx = np.random.randint(0, len(arr))
    return arr[idx]


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ImageNet', choices=['ImageNet', 'CIFAR10', 'MNIST'])
parser.add_argument('--model', type=str, default='resnet50',
                    choices=['resnet50', 'vgg16_bn', 'mobilenet_v2', 'inception_v3', 'densenet121', 'LeNet1', 'LeNet5'])
parser.add_argument('--op', type=str, default='all', choices=['all', 'G', 'P', 'S', 'A', 'W'])
parser.add_argument('--output_root', type=str, default='')
parser.add_argument('--nc', type=int, default=3)
parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=0)

args = parser.parse_args()

# 加载模型和参数
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# model.to(device)

'''
# DeepLift
model_ex = copy.deepcopy(model)
dl = DeepLift(model_ex)

# DeepLift Example
# dl = DeepLift(model_ex)
# for i, (image, label, prefix) in enumerate(tqdm(test_loader)):
#     print("loop")
#     image.requires_grad = True
#     baselines = torch.zeros_like(image)
#     logit = model(image)
#     pred = logit.max(-1)
#     attr = dl.attribute(image, target=pred, baselines=baselines)
#     th, closing, opening = attr2concept(attr[0])
'''

# 加载数据集
if args.dataset == 'ImageNet':
    train_dataset = ImageNetDataset(args, split='train')
    test_dataset = ImageNetDataset(args, split='val')
elif args.dataset == 'CIFAR10':
    train_dataset = CIFAR10Dataset(args, split='train')
    test_dataset = CIFAR10Dataset(args, split='test')
elif args.dataset == 'MNIST':
    train_dataset = MNISTDataset(args, split='train')
    test_dataset = MNISTDataset(args, split='test')

train_loader = DataLoader(args).get_loader(dataset=train_dataset, shuffle=True)
test_loader = DataLoader(args).get_loader(dataset=test_dataset, shuffle=True)


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

with open('imagenet_class_index.json', 'r', encoding='utf-8') as fp:
    imagenet_class_index = json.load(fp)
fp.close()


model.eval()
dataiter = iter(test_loader)


# 算法
mutation_strategy = mutation(model, step=0.5, itr_num=50)

with torch.no_grad():
    for i in range(1):
        images, labels = dataiter.next()
        # images = images.to(device)

        # Mutation模块
        mutated_img, coverage_rate = mutation_strategy.generate(images)

        # 输出新旧标签
        _, labels = torch.max(model(images), 1)
        _, mutated_labels = torch.max(model(mutated_img), 1)
        label = imagenet_class_index[str(int(labels))][1]
        mutated_label = imagenet_class_index[str(int(mutated_labels))][1]
        print('Original label is: ', '%s' % label)
        print('Mutated label is: ', '%s' % mutated_label)

        # 文件保存，文件名为旧label-新label
        gen_img = Image.fromarray(to_image(mutated_img))
        image_name = label + "---" + mutated_label + ".JPEG"
        gen_img.save(f"./gen_input/{image_name}")
