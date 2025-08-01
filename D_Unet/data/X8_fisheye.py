import math
import os
import random
import PIL
import cv2
import numpy as np
import torchvision
import torch

from torch.utils.data import Dataset
from PIL import Image
from os import listdir


def data_transform(X):
    return 2 * X - 1.0


def compute_map_ws(img):
    equ = torch.zeros((img.shape[0], img.shape[1]))
    for i in range(0, equ.shape[0]):
        equ[i, :] = genERP(i, equ.shape[0])
    return equ


def genERP(j, N):
    val = math.pi / N
    w = math.cos((j - (N / 2) + 0.5) * val)
    return w


def random_crops_tensor(img_tensor, y, x, h, w):
    new_crop = img_tensor[:, y:y + h, x:x + w]  # 先行在列# 根据位置随机裁剪
    return new_crop


class oditra(Dataset):
    def __init__(self, dictC):
        params = dictC['params']
        db_dir = params.db_dir
        size = params.HR_size
        LR_names, BIC_names, HR_names = [], [], []

        inputs_LR = os.path.join(db_dir, 'crop_LR_fisheye', 'X4')
        inputs_BIC = os.path.join(db_dir, 'crop_LR_bicubic_fisheye', 'X4')
        inputs_HR = os.path.join(db_dir, 'crop_HR')
        images_BIC = [f for f in listdir(inputs_BIC)]  # 遍历 odi360_inputs 目录下的所有文件和文件夹，并将它们的名称添加到 images 列表中。
        images_LR = [f for f in listdir(inputs_LR)]
        images_HR = [f for f in listdir(inputs_HR)]
        print(len(images_BIC))
        print(len(images_LR))
        print(len(images_HR))
        BIC_names += [os.path.join(inputs_BIC, i) for i in images_BIC]
        LR_names += [os.path.join(inputs_LR, i) for i in images_LR]
        HR_names += [os.path.join(inputs_HR, i) for i in images_HR]
        x = list(enumerate(LR_names))
        random.shuffle(x)
        indices, input_LR_names = zip(*x)  # 将随机排序后的 x 列表解压缩为 indices 和 input_names 两个列表
        input_BIC_names = [BIC_names[idx] for idx in indices]
        input_HR_names = [HR_names[idx] for idx in indices]

        matrix_128 = torch.zeros((128, 256))
        matimg_128 = compute_map_ws(matrix_128)
        matimg_128 = matimg_128.unsqueeze(0)

        matrix_256 = torch.zeros((256, 512))
        matimg_256 = compute_map_ws(matrix_256)
        matimg_256 = matimg_256.unsqueeze(0)

        matrix_512 = torch.zeros((512, 1024))
        matimg_512 = compute_map_ws(matrix_512)
        matimg_512 = matimg_512.unsqueeze(0)

        matrix_1024 = torch.zeros((1024, 2048))
        matimg_1024 = compute_map_ws(matrix_1024)
        matimg_1024 = matimg_1024.unsqueeze(0)

        self.matrix_128 = matimg_128
        self.matrix_256 = matimg_256
        self.matrix_512 = matimg_512
        self.matrix_1024 = matimg_1024
        self.size = size
        self.input_BIC_names = input_BIC_names
        self.input_LR_names = input_LR_names
        self.input_HR_names = input_HR_names
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])  # 建一个 torchvision 的图像转换管道（transforms pipeline）,将 PIL 图像对象转换为 PyTorch 张量。

    def __getitem__(self, index):
        input_BIC_names = self.input_BIC_names[index]
        input_LR_names = self.input_LR_names[index]
        input_HR_names = self.input_HR_names[index]
        part = input_LR_names.split('_')
        x = int(part[-2])
        y = int(part[-1].split('.')[0])
        input_BIC_img  = PIL.Image.open(os.path.join(input_BIC_names))
        input_LR_img = PIL.Image.open(os.path.join(input_LR_names))
        input_HR_img = PIL.Image.open(os.path.join(input_HR_names))

        matrix_list = [
            (random_crops_tensor(self.matrix_128, int(y / 8), int(x / 8), int(self.size / 8), int(self.size / 8))),
            (random_crops_tensor(self.matrix_256, int(y / 4), int(x / 4), int(self.size / 4), int(self.size / 4))),
            (random_crops_tensor(self.matrix_512, int(y / 2), int(x / 2), int(self.size / 2), int(self.size / 2))),
            (random_crops_tensor(self.matrix_1024, int(y), int(x), self.size, self.size))
        ]
        lr_bic = data_transform(self.transforms(input_BIC_img))
        lr = data_transform(self.transforms(input_LR_img))
        hr = data_transform(self.transforms(input_HR_img))
        return {'HR': hr, 'LR': lr, 'LR_bic': lr_bic, 'matrix_list':matrix_list}

    def __len__(self):
        return len(self.input_HR_names)

class odival(Dataset):
    def __init__(self, dictC):
        params = dictC['params']
        db_dir = params.db_dir
        size = params.HR_size
        LR_names, BIC_names, HR_names = [], [], []

        inputs_LR = os.path.join(db_dir, 'LR_fisheye', 'X4')
        inputs_BIC = os.path.join(db_dir, 'LR_bicubic_fisheye', 'X4')
        inputs_HR = os.path.join(db_dir, 'HR')
        images_BIC = [f for f in listdir(inputs_BIC)]  # 遍历 odi360_inputs 目录下的所有文件和文件夹，并将它们的名称添加到 images 列表中。
        images_LR = [f for f in listdir(inputs_LR)]
        images_HR = [f for f in listdir(inputs_HR)]
        print(len(images_BIC))
        print(len(images_LR))
        print(len(images_HR))
        BIC_names += [os.path.join(inputs_BIC, i) for i in images_BIC]
        LR_names += [os.path.join(inputs_LR, i) for i in images_LR]
        HR_names += [os.path.join(inputs_HR, i) for i in images_HR]
        x = list(enumerate(LR_names))
        indices, input_LR_names = zip(*x)  # 将随机排序后的 x 列表解压缩为 indices 和 input_names 两个列表
        input_BIC_names = [BIC_names[idx] for idx in indices]
        input_HR_names = [HR_names[idx] for idx in indices]

        matrix_128 = torch.zeros((128, 256))
        matimg_128 = compute_map_ws(matrix_128)
        matimg_128 = matimg_128.unsqueeze(0)

        matrix_256 = torch.zeros((256, 512))
        matimg_256 = compute_map_ws(matrix_256)
        matimg_256 = matimg_256.unsqueeze(0)

        matrix_512 = torch.zeros((512, 1024))
        matimg_512 = compute_map_ws(matrix_512)
        matimg_512 = matimg_512.unsqueeze(0)

        matrix_1024 = torch.zeros((1024, 2048))
        matimg_1024 = compute_map_ws(matrix_1024)
        matimg_1024 = matimg_1024.unsqueeze(0)

        self.matrix_128 = matimg_128
        self.matrix_256 = matimg_256
        self.matrix_512 = matimg_512
        self.matrix_1024 = matimg_1024
        self.size = size
        self.input_BIC_names = input_BIC_names
        self.input_LR_names = input_LR_names
        self.input_HR_names = input_HR_names
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])  # 建一个 torchvision 的图像转换管道（transforms pipeline）,将 PIL 图像对象转换为 PyTorch 张量。

    def __getitem__(self, index):
        input_BIC_names = self.input_BIC_names[index]
        input_LR_names = self.input_LR_names[index]
        input_HR_names = self.input_HR_names[index]
        part = input_LR_names.split('_')
        x = int(part[-2])
        y = int(part[-1].split('.')[0])
        input_BIC_img  = PIL.Image.open(os.path.join(input_BIC_names))
        input_LR_img = PIL.Image.open(os.path.join(input_LR_names))
        input_HR_img = PIL.Image.open(os.path.join(input_HR_names))

        matrix_list = [
            (random_crops_tensor(self.matrix_128, int(y / 8), int(x / 8), int(self.size / 8), int(self.size / 8))),
            (random_crops_tensor(self.matrix_256, int(y / 4), int(x / 4), int(self.size / 4), int(self.size / 4))),
            (random_crops_tensor(self.matrix_512, int(y / 2), int(x / 2), int(self.size / 2), int(self.size / 2))),
            (random_crops_tensor(self.matrix_1024, int(y), int(x), self.size, self.size))
        ]
        lr_bic = data_transform(self.transforms(input_BIC_img))
        lr = data_transform(self.transforms(input_LR_img))
        hr = data_transform(self.transforms(input_HR_img))
        return {'HR': hr, 'LR': lr, 'LR_bic': lr_bic, 'matrix_list':matrix_list}

    def __len__(self):
        return len(self.input_HR_names)

class oditest(Dataset):
    def __init__(self, dictC):
        params = dictC['params']
        db_dir = params['db_dir']
        LR_names, HR_names = [], []

        inputs_LR = os.path.join(db_dir, 'LR_ERP', 'X4')
        images_PNG = [f for f in listdir(inputs_LR)]  # 遍历 odi360_inputs 目录下的所有文件和文件夹，并将它们的名称添加到 images 列表中。
        print(len(images_PNG))
        LR_names += [os.path.join(inputs_LR, i) for i in images_PNG]
        x = list(enumerate(LR_names))
        indices, input_LR_names = zip(*x)  # 将随机排序后的 x 列表解压缩为 indices 和 input_names 两个列表

        matrix_128 = torch.zeros((128, 256))
        matimg_128 = compute_map_ws(matrix_128)
        matimg_128 = matimg_128.unsqueeze(0)

        matrix_256 = torch.zeros((256, 512))
        matimg_256 = compute_map_ws(matrix_256)
        matimg_256 = matimg_256.unsqueeze(0)

        matrix_512 = torch.zeros((512, 1024))
        matimg_512 = compute_map_ws(matrix_512)
        matimg_512 = matimg_512.unsqueeze(0)

        matrix_1024 = torch.zeros((1024, 2048))
        matimg_1024 = compute_map_ws(matrix_1024)
        matimg_1024 = matimg_1024.unsqueeze(0)

        self.matrix_128 = matimg_128
        self.matrix_256 = matimg_256
        self.matrix_512 = matimg_512
        self.matrix_1024 = matimg_1024
        self.input_LR_names = input_LR_names
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])  # 建一个 torchvision 的图像转换管道（transforms pipeline）,将 PIL 图像对象转换为 PyTorch 张量。

    def __getitem__(self, index):
        input_LR_names = self.input_LR_names[index]
        img_id = os.path.basename(input_LR_names)
        input_LR_img = PIL.Image.open(os.path.join(input_LR_names))
        lr_bic = cv2.resize(np.array(input_LR_img), (2048, 1024), interpolation=cv2.INTER_CUBIC)
        matrix_list = [self.matrix_128, self.matrix_256, self.matrix_512, self.matrix_1024]
        lr = data_transform(self.transforms(input_LR_img))
        lr_bic = data_transform(self.transforms(lr_bic))
        return {'LR': lr, 'LR_bic': lr_bic, 'matrix_list':matrix_list}, img_id

    def __len__(self):
        return len(self.input_LR_names)


