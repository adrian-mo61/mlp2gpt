# model.py
import torch
from torch import nn
import torch.nn.functional as F

class MnistModel(nn.Module):
    def __init__(self,input_size,hidden_size,output_size, dropout_rate=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_size,hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0],hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1],output_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self,x):

        x = torch.flatten(x,1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        x = self.fc3(x)
        return x

