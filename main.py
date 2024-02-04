import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms

#create CNN model
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1,out_channels = 10, kernel_size=(3,3),stride=(1,1), padding = (1,1))
        self.pool = nn.MaxPool2d(kernal_size = (2,2), stride = (2,2))
        self.conv2 = nn.Conv2s(in_channels = 10,out_channels = 10, kernal_size=(3,3), stride=(1,1), padding=(1,1))
       ## self.fc1 = nn.Linear(,num_classes)