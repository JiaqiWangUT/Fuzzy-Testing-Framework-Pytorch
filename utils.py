# general
import itertools
import random
import os
import numpy as np
from PIL import Image
import math
import json
import cv2
import shutil

# torch
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision

def get_img(dir_path):
    while True:
        # orig_img为随机选择的图片文件名
        orig_img = random.choice(os.listdir(dir_path))
        if orig_img == ".git":
            continue
        img = Image.open(dir_path + orig_img)
        img = img.resize((224, 224), Image.LANCZOS)
        return transforms.ToTensor()(img).reshape(-1, 3, 224, 224)


def to_image(img_tensor):
    img = img_tensor.squeeze(0).detach()
    img = img.transpose(0, 2).transpose(0, 1).numpy()
    img[..., 0] *= 255
    img[..., 1] *= 255
    img[..., 2] *= 255
    img = np.clip(img, 0, 255).astype("uint8")
    return img


def constraint_light(grad):
    return 1e4 * grad.mean() * torch.ones_like(grad)


def constraint_black(grads, rect_shape=(10, 10)):
    start_point = (
        random.randint(0, grads.shape[2] - rect_shape[0]), random.randint(0, grads.shape[3] - rect_shape[1]))
    new_grads = torch.zeros_like(grads)
    patch = grads[..., start_point[0]:start_point[0] + rect_shape[0], start_point[1]:start_point[1] + rect_shape[1]]
    if torch.mean(patch) < 0:
        new_grads[..., start_point[0]:start_point[0] + rect_shape[0],
                  start_point[1]:start_point[1] + rect_shape[1]] = -torch.ones_like(patch)
    return new_grads


def constraint_occl(grads, start_point, rect_shape):
    new_grads = torch.zeros_like(grads)
    new_grads[..., start_point[0]:start_point[0] + rect_shape[0],
              start_point[1]:start_point[1] + rect_shape[1]] = grads[..., start_point[0]:start_point[0] + rect_shape[0],
                                                                     start_point[1]:start_point[1] + rect_shape[1]]
    return new_grads


def mask_ratio(src):
    src_mask = src == 255
    ones = np.ones(src.shape
                   )
    return src_mask.astype(np.float32).sum() / ones.sum()


def IoU(src, tgt):
    src_mask = src == 255
    tgt_mask = tgt == 255
    return (src_mask & tgt_mask).astype(np.float32).sum() / (src_mask | tgt_mask).astype(np.float32).sum()


def segment_org(org_tensor, mask):
    # print('org_tensor: ', org_tensor.size())
    if isinstance(org_tensor, torch.Tensor):
        org_np = org_tensor.squeeze().cpu().detach().numpy()
    else:
        org_np = org_tensor
    # print(mask.max())
    for i in range(3):
        org_np[i] *= (mask == 255).astype(np.uint8)
    org_np = np.transpose(org_np, (1, 2, 0))
    # print('org_np: ', org_np.max())
    # org_np = cv2.cvtColor(org_np, cv2.COLOR_RGB2BGR)
    return org_np * 255


def segment_org_white(org_tensor, mask):
    # print('org_tensor: ', org_tensor.size())
    if isinstance(org_tensor, torch.Tensor):
        org_np = org_tensor.squeeze().cpu().detach().numpy()
    else:
        org_np = org_tensor
    # print(mask.max())
    for i in range(3):
        org_np[i] *= (mask == 255).astype(np.uint8)
        org_np[i] += (1 * (mask != 255)).astype(np.uint8)
    org_np = np.transpose(org_np, (1, 2, 0))
    # print('org_np: ', org_np.max())
    # org_np = cv2.cvtColor(org_np, cv2.COLOR_RGB2BGR)
    return org_np * 255


def segment_org_green(org_tensor, mask):
    # print('org_tensor: ', org_tensor.size())
    if isinstance(org_tensor, torch.Tensor):
        org_np = org_tensor.squeeze().cpu().detach().numpy()
    else:
        org_np = org_tensor
    # print(mask.max())
    for i in range(3):
        org_np[i] *= (mask == 255).astype(np.uint8)
    org_np[1] += (1 * (mask != 255)).astype(np.uint8)
    org_np = np.transpose(org_np, (1, 2, 0))
    # print('org_np: ', org_np.max())
    # org_np = cv2.cvtColor(org_np, cv2.COLOR_RGB2BGR)
    return org_np * 255


def attr2concept(attr):
    if isinstance(attr, torch.Tensor):
        pixel0 = attr.squeeze().cpu().detach().numpy()
    else:
        pixel0 = attr
    index = np.where(pixel0 > 0)[0]
    pixel0_pos = pixel0[index]
    pixel0_pos = pixel0_pos.mean(0)
    pixel0_pos -= pixel0_pos.min()
    pixel0_pos /= pixel0_pos.max()
    pixel0_pos *= 255
    pixel0_pos = pixel0_pos.astype(np.uint8)
    # _, th = cv2.threshold(pixel0_pos, int(255 * 0.6), 255, cv2.THRESH_BINARY)
    _, th = cv2.threshold(pixel0_pos, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    kernel2 = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel2)
    return th, closing, opening


def attr2concept_mnist(attr):
    if isinstance(attr, torch.Tensor):
        pixel0 = attr.cpu().detach().numpy()
    else:
        pixel0 = attr
    index = np.where(pixel0 > 0)[0]
    pixel0_pos = pixel0[index]
    pixel0_pos = pixel0_pos.mean(0)
    pixel0_pos -= pixel0_pos.min()
    pixel0_pos /= pixel0_pos.max()
    pixel0_pos *= 255
    pixel0_pos = pixel0_pos.astype(np.uint8)
    # _, th = cv2.threshold(pixel0_pos, int(255 * 0.7), 255, cv2.THRESH_BINARY)
    _, th = cv2.threshold(pixel0_pos, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    kernel2 = np.ones((2, 2), np.uint8)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel2)
    # return th, closing, opening
    return th, closing, closing


def image_normalize(image, dataset):
    if dataset == 'CIFAR10':
        transform = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
    elif dataset == 'ImageNet':
        transform = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    elif dataset == 'MNIST':
        transform = transforms.Normalize((0.1307,), (0.3081,))
    else:
        raise NotImplementedError
    return transform(image)


def accuracy(scores, targets, k=1):
    """
    Computes top-k accuracy, from predicted and true labels.
    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    # print('bs:', batch_size)
    _, ind = scores.topk(k, dim=1, largest=True, sorted=True)
    # print('ind: ', ind.shape)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    # print('correct: ', correct.shape)
    # print(correct)
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() / batch_size


def make_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


def load_params(json_file):
    with open(json_file) as f:
        return json.load(f)


def get_batch(data_loader):
    while True:
        for batch in data_loader:
            yield batch


