import torch
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import torch
import torch.nn as nn
import math
import os
import torchvision.datasets as dset

# data
image_folder_path = "../mbti/"

def gain_sample(dataset, batch_size, image_size=64): # 원래 4

    batch_size = 64   
    dataset = dset.ImageFolder(root=image_folder_path,
                           transform=transforms.Compose([ # 전처리 작업 
                               transforms.Scale(image_size), # 이미지 크기 64로 조정 
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 컬러 값이라 채널 3개를 사용  
                           ]))

    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    return loader

dataset        = datasets.ImageFolder(image_folder_path)
origin_loader  = gain_sample(dataset, 240, 64)
data_loader    = iter(origin_loader)

for i in range(10):

    real_image, label = next(data_loader)
    torchvision.utils.save_image(real_image, f'../previews/ESTJ/m/preview{i}.png', nrow=24, padding=2, normalize=True, range=(-1,1))


