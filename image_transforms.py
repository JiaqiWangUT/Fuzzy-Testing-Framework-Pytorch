import numpy as np
import cv2
import torch
import torch.nn.functional as F
import math


# def pad(src, dst):
#     H_src, W_src, C_src = src.shape
#     H_dst, W_dst, C_dst = dst.shape
#     top = np.max((H_src - H_dst) // 2, 0)
#     bottom = top
#     left = np.max((W_src - W_dst) // 2, 0)
#     right = left
#     COLOR = [255, 255, 255] # white
#     padded = cv2.copyMakeBorder(dst, top, bottom, left, right, cv2.BORDER_CONSTANT, value=COLOR)
#     return padded

def pad(src, dst):
    # pad with white pixel
    H_src, W_src, C = src.shape
    H_dst, W_dst = dst.shape[:2]

    top = np.max((H_src - H_dst) // 2, 0)
    left = np.max((W_src - W_dst) // 2, 0)

    if len(dst.shape) == 2:
        COLOR = 255
        padded = np.full((H_src, W_src, C), COLOR, dtype=np.uint8)
        padded[top:top + H_dst, left:left + W_dst, 0] = dst
    else:
        COLOR = (255, 255, 255)
        padded = np.full((H_src, W_src, C), COLOR, dtype=np.uint8)
        padded[top:top + H_dst, left:left + W_dst] = dst
    return padded


# def pad(src, dst):
#     # pad with black pixel
#     H_src, W_src, C = src.shape
#     H_dst, W_dst = dst.shape[:2]

#     top = np.max((H_src - H_dst) // 2, 0)
#     left = np.max((W_src - W_dst) // 2, 0)

#     if len(dst.shape) == 2:
#         COLOR = 0
#         padded = np.full((H_src, W_src, C), COLOR, dtype=np.uint8)
#         padded[top:top+H_dst, left:left+W_dst, 0] = dst
#     else:
#         COLOR = (0, 0, 0)
#         padded = np.full((H_src, W_src, C), COLOR, dtype=np.uint8)
#         padded[top:top+H_dst, left:left+W_dst] = dst
#     return padded


def image_translation(img, params):
    # params(x,y),范围-2~2,x为横向平移,y为纵向平移,左上为正,右下为负
    img = img.resize(3, 224, 224)
    theta = torch.tensor([[1., 0., params[0]], [0., 1., params[1]]], dtype=torch.float)
    grid = F.affine_grid(theta.unsqueeze(0), img.unsqueeze(0).size())
    output = F.grid_sample(img.unsqueeze(0), grid)
    return output.resize(1, 3, 224, 224)


def image_scale(img, params):
    # params大于0,大于1为缩小,小于1为放大
    img = img.resize(3, 224, 224)
    theta = torch.tensor([[params, 0., 0], [0., params, 0]], dtype=torch.float)
    grid = F.affine_grid(theta.unsqueeze(0), img.unsqueeze(0).size())
    output = F.grid_sample(img.unsqueeze(0), grid)
    # output = pad(img, output)
    return output.resize(1, 3, 224, 224)


def image_shear(img, params):
    img = img.resize(3, 224, 224)
    theta = torch.tensor([[1., params, 0], [0., 1., 0]], dtype=torch.float)
    grid = F.affine_grid(theta.unsqueeze(0), img.unsqueeze(0).size())
    output = F.grid_sample(img.unsqueeze(0), grid)
    # output = pad(img, output)
    return output.resize(1, 3, 224, 224)


def image_rotation(img, params):
    # params为旋转角度
    params = math.radians(params)
    img = img.resize(3, 224, 224)
    theta = torch.tensor([[math.cos(params), math.sin(params), 0], [-math.sin(params), math.cos(params), 0]],
                         dtype=torch.float)
    grid = F.affine_grid(theta.unsqueeze(0), img.unsqueeze(0).size())
    output = F.grid_sample(img.unsqueeze(0), grid)
    return output.resize(1, 3, 224, 224)


def image_contrast(img, params):
    alpha = params
    img_np = img.numpy() * 255
    new_img = cv2.convertScaleAbs(img_np, beta=0, alpha=alpha) / 255.0
    new_img = torch.tensor(new_img)
    return new_img.to(torch.float32)


def image_brightness(img, params):
    beta = params
    img_np = img.numpy() * 255
    new_img = cv2.convertScaleAbs(img_np, beta=beta, alpha=1) / 255.0
    new_img = torch.tensor(new_img)
    return new_img.to(torch.float32)


def image_blur(img, params):
    img_np = img.numpy().reshape(3, 224, 224)
    img_type = img_np.dtype
    if (np.issubdtype(img_type, np.integer)):
        img_np = np.uint8(img_np)
    else:
        img_np = np.float32(img_np)

    blur = []
    if params == 1:
        blur = cv2.blur(img_np, (3, 3))
    if params == 2:
        blur = cv2.blur(img_np, (4, 4))
    if params == 3:
        blur = cv2.blur(img_np, (5, 5))
    if params == 4:
        blur = cv2.GaussianBlur(img_np, (3, 3), 0)
    if params == 5:
        blur = cv2.GaussianBlur(img_np, (5, 5), 0)
    if params == 6:
        blur = cv2.GaussianBlur(img_np, (7, 7), 0)
    if params == 7:
        blur = cv2.medianBlur(img_np, 3)
    if params == 8:
        blur = cv2.medianBlur(img_np, 5)
    if params == 9:
        blur = cv2.blur(img_np, (6, 6))
    if params == 10:
        blur = cv2.bilateralFilter(img_np, 9, 75, 75)

    blur = blur.astype(img_type).reshape(1, 3, 224, 224)
    new_img = torch.tensor(blur)
    return new_img.to(torch.float32)
