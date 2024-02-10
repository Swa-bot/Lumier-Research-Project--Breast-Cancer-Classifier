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
        self.conv1 = nn.Conv2d(1,1024,kernel_size=3)
        self.conv2=nn.Conv2d(1024,512,kernel_size=3)
        self.conv3 = nn.Conv2d(512,256,kernel_size=3)
        self.conv4 = nn.Conv2d(256,128,kernel_size=3)
        self.linear = nn.Linear(56,2)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x=self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.linear(x)
        x=self.sigmoid(x)

        return x


    
#Initialize Network
model = NN(input_size=input_size,num_classes=num_classes).to(device)
#Loss & Optimizer
criterion = nn.BCELoss(reduction='sum')
optimizer =  optim.Adam(model.parameters(),lr=learning_rate)
#Train Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets=targets.to(device=device)

        print(data.shape)
        data.reshape(data.shape[0],-1)

        scores = model(data)
        targets = targets.unsqueeze(1)
        targets = targets.unsqueeze(1)
        print(scores.shape)
        print(targets.shape)
        loss = criterion(scores,targets)

        print('Epoch',epoch,'Loss:',loss.item(), '- Pred:',scores.data[0])

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

#Check Accuracy
'''def checkAccuracy(loader,model):
    num_correct = 0
    num_tests = 0

    model.eval()

    with torch.no_grad():
        for x,y in loader:
            x = x.to(device=device)
            y = y.to(device.device)

            x=x.reshape(x.shape[0],-1)
            scores = model.x()

            predictions = scores.max(1)
            num_correct+= (predictions==y).sum()
            num_tests += predictions.size(0)
            
'''


