# data.py
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from modelscope import MsDataset


# 归一化
transform = transforms.Compose([
    transforms.ToTensor(),          # 转为 [0,1] 的 Tensor
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 均值和标准差
])

class MnistDataset(Dataset):
    def __init__(self,ms_dataset,transform=None):
        self.ms_dataset = ms_dataset
        self.transform = transform
    def __len__(self):
        return len(self.ms_dataset)
    def __getitem__(self,idx):
        sample = self.ms_dataset[idx]
        image = sample['image']
        label = sample['label']

        if self.transform:
            image = self.transform(image)
        return image,label

