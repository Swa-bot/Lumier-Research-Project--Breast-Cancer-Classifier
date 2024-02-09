#imports
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from medmnist import BreastMNIST

#Hyperparameters
train_batch_size = 64
test_batch_size = 10
num_classes = 2
learning_rate = 0.001
input_size = 784
num_epochs = 1

#Load Dataset
train_dataset = BreastMNIST(split = 'train', transform=transforms.ToTensor(),download=True,size=64,root="dataset/")
train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size,shuffle=True)

test_dataset = BreastMNIST(split="test",transform=transforms.ToTensor(),download=True,size=64,root='dataset/' )
test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size,shuffle=True)

#Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Build CNN
class NN(nn.Module):
    def __init__(self,input_size,num_classes):
        super(NN,self).__init__()
        self.fc1 = nn.Linear(input_size,50)
        self.fc2 = nn.Linear(50,num_classes)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
#Initialize Network
model = NN(input_size=input_size,num_classes=num_classes).to(device)
#Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)
#Train Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targers=targets.to(device=device)
        print(data.shape)

