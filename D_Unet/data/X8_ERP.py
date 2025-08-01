import math
import os
import random
import PIL
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
    new_crop = img_tensor[:,  y:y + h, x:x + w]  # 先行在列# 根据位置随机裁剪
    return new_crop


class oditra(Dataset):
    def __init__(self, dictC):
        params = dictC['params']
        db_dir = params.db_dir
        LR_names, BIC_names, HR_names = [], [], []

        inputs_LR = os.path.join(db_dir, 'crop_LR_ERP', 'X8')
        inputs_BIC = os.path.join(db_dir, 'crop_LR_bicubic_ERP', 'X8')
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

        matrix_lr = torch.zeros((128, 256))
        matimg_lr = compute_map_ws(matrix_lr)
        matimg_lr = matimg_lr.unsqueeze(0)

        matrix_hr = torch.zeros((1024, 2048))
        matimg_hr = compute_map_ws(matrix_hr)
        matimg_hr = matimg_hr.unsqueeze(0)

        self.matrix_lr = matimg_lr
        self.matimg_hr = matimg_hr
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

        matrix_img_lr = random_crops_tensor(self.matrix_lr, int(y / 8), int(x / 8), 32, 32)
        matrix_img_hr = random_crops_tensor(self.matimg_hr, int(y), int(x), 256, 256)
        lr_bic = data_transform(self.transforms(input_BIC_img))
        lr = data_transform(self.transforms(input_LR_img))
        hr = data_transform(self.transforms(input_HR_img))
        return {'HR': hr, 'LR': lr, 'LR_bic': lr_bic, 'Mat_LR': matrix_img_lr, 'Mat_HR': matrix_img_hr}

    def __len__(self):
        return len(self.input_HR_names)

class odival(Dataset):
    def __init__(self, dictC):
        params = dictC['params']
        db_dir = params.db_dir
        LR_names, BIC_names, HR_names = [], [], []

        inputs_LR = os.path.join(db_dir, 'LR_fisheye', 'X8')
        inputs_BIC = os.path.join(db_dir, 'LR_bicubic_fisheye', 'X8')
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

        matrix_lr = torch.zeros((128, 256))
        matimg_lr = compute_map_ws(matrix_lr)
        matimg_lr = matimg_lr.unsqueeze(0)

        matrix_hr = torch.zeros((1024, 2048))
        matimg_hr = compute_map_ws(matrix_hr)
        matimg_hr = matimg_hr.unsqueeze(0)

        self.matrix_lr = matimg_lr
        self.matimg_hr = matimg_hr
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

        matrix_img_lr = random_crops_tensor(self.matrix_lr, int(y / 8), int(x / 8), 32, 32)
        matrix_img_hr = random_crops_tensor(self.matimg_hr, int(y), int(x), 256, 256)
        lr_bic = data_transform(self.transforms(input_BIC_img))
        lr = data_transform(self.transforms(input_LR_img))
        hr = data_transform(self.transforms(input_HR_img))
        return {'HR': hr, 'LR': lr, 'LR_bic': lr_bic, 'Mat_LR': matrix_img_lr, 'Mat_HR': matrix_img_hr}

    def __len__(self):
        return len(self.input_HR_names)

class oditest(Dataset):
    def __init__(self, dictC):
        params = dictC['params']
        db_dir = params['db_dir']
        LR_names, BIC_names, HR_names = [], [], []

        inputs_LR = os.path.join(db_dir, 'LR_fisheye', 'X8')
        inputs_BIC = os.path.join(db_dir, 'LR_bicubic_fisheye', 'X8')
        images_PNG = [f for f in listdir(inputs_LR)]  # 遍历 odi360_inputs 目录下的所有文件和文件夹，并将它们的名称添加到 images 列表中。
        print(len(images_PNG))
        BIC_names += [os.path.join(inputs_BIC, i) for i in images_PNG]
        LR_names += [os.path.join(inputs_LR, i) for i in images_PNG]
        x = list(enumerate(LR_names))
        indices, input_LR_names = zip(*x)  # 将随机排序后的 x 列表解压缩为 indices 和 input_names 两个列表
        input_BIC_names = [BIC_names[idx] for idx in indices]

        matrix_lr = torch.zeros((128, 256))
        matimg_lr = compute_map_ws(matrix_lr)
        matimg_lr = matimg_lr.unsqueeze(0)

        matrix_hr = torch.zeros((1024, 2048))
        matimg_hr = compute_map_ws(matrix_hr)
        matimg_hr = matimg_hr.unsqueeze(0)

        self.matrix_lr = matimg_lr
        self.matimg_hr = matimg_hr
        self.input_BIC_names = input_BIC_names
        self.input_LR_names = input_LR_names
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])  # 建一个 torchvision 的图像转换管道（transforms pipeline）,将 PIL 图像对象转换为 PyTorch 张量。

    def __getitem__(self, index):
        input_LR_BIC_names = self.input_BIC_names[index]
        input_LR_names = self.input_LR_names[index]
        img_id = os.path.basename(input_LR_names)
        input_LR_BIC_img = PIL.Image.open(os.path.join(input_LR_BIC_names))
        input_LR_img = PIL.Image.open(os.path.join(input_LR_names))

        matrix_img_lr = self.matrix_lr
        matrix_img_hr = self.matimg_hr
        lr_bic = data_transform(self.transforms(input_LR_BIC_img))
        lr = data_transform(self.transforms(input_LR_img))
        return {'LR': lr, 'LR_bic': lr_bic, 'Mat_LR': matrix_img_lr, 'Mat_HR': matrix_img_hr}, img_id

    def __len__(self):
        return len(self.input_LR_names)


