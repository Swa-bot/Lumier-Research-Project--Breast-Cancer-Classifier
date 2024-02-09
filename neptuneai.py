import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
import torch.optim as optim
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import os

data_dir = "/content/gdrive/MyDrive"
folder_name = "kaggle"
image_folders = os.path.join(data_dir, folder_name)

transform = transforms.Compose([transforms.Resize((50, 50)), transforms.ToTensor()])
images = []
for file in os.listdir(image_folders):
    try:
      images.append(ImageFolder(os.path.join(image_folders, file), transform=transform))
    except:
      print(file)
datasets = torch.utils.data.ConcatDataset(images)

i=0
for dataset in datasets.datasets:
    if i==0:
        result = Counter(dataset.targets)
        i += 1
    else:
        result += Counter(dataset.targets)

result = dict(result)
print("""Total Number of Images for each Class:
    Class 0 (No Breast Cancer): {}
    Class 1 (Breast Cancer present): {}""".format(result[0], result[1]))

random_seed = 42
torch.manual_seed(random_seed)

test_size = int(0.25*(result[0]+result[1]))
print(test_size)
train_size = len(datasets) - test_size
train_dataset, test_dataset = random_split(datasets, [train_size, test_size])

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128,
                                          shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64,
                                         shuffle=False, num_workers=2)


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()
# show images
imshow(torchvision.utils.make_grid(images[:6], nrow=3))
# show labels
labels[:6]

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

class BreastCancerClassifyNet(nn.Module):
  def __init__(self):
    super(BreastCancerClassifyNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
    self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
    self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
    self.pool = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(4096, 1024)
    self.fc2 = nn.Linear(1024, 512)
    self.fc3 = nn.Linear(512, 1)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = self.pool(F.relu(self.conv3(x)))
    x = x.view(-1, self.flat_features(x))
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    x = F.log_softmax(x)
    return x

  def flat_features(self, x):
    size = x.size()[1:]
    num_features = 1
    for s in size:
      num_features *= s
    return num_features

net = BreastCancerClassifyNet()
net = net.to(device)

criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001)

test_data_iter = iter(testloader)
test_images, test_labels = test_data_iter.next()
for epoch in range(20):
  running_loss = 0
  for i, data in enumerate(trainloader, 0):
    input_imgs, labels = data
    input_imgs = input_imgs.to(device)
    labels = labels.to(device)
    optimizer.zero_grad()
    outputs = net(input_imgs)
    labels = labels.unsqueeze(1).float()
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    #printing stats and checking prediction as we train
    running_loss += loss.item()
    if i % 10000 == 0:
      print('epoch', epoch+1, 'loss', running_loss/10000)
      imshow(torchvision.utils.make_grid(test_images[0].detach()))
      test_out = net(test_images.to(device))
      _, predicted_out = torch.max(test_out, 1)
      print('Predicted : ', ' '.join('%5s' % predicted_out[0]))
print('Training finished')

correct = 0
total = 0
with torch.no_grad():
  for data in testloader:
    test_images, test_labels = data
    test_out = net(test_images.to(device))
    _, predicted = torch.max(test_out.data, 1)
    total += test_labels.size(0)
    for _id, out_pred in enumerate(predicted):
      if int(out_pred) == int(test_labels[_id]):
        correct += 1

print('Accuracy of the network on the 44252 test images: %d %%' % (
        100 * correct / total))