# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import helper
from torch.nn import Conv2d, functional as F, Linear, MaxPool2d, Module
from torch import nn as nn
import torchvision
import pandas as pd
import numpy as np
from torch.utils.data import random_split

transform = transforms.Compose([transforms.Resize(128),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                                

train_dataset = datasets.ImageFolder('./melspectro_data/train', transform=transform)
test_dataset = datasets.ImageFolder('./melspectro_data/test', transform=transform)

# split train into valid and train
train_size = int(0.8 * len(train_dataset))
valid_size = len(train_dataset) - train_size
partial_train_ds, valid_ds = random_split(train_dataset, [train_size, valid_size])

train_dataloader = torch.utils.data.DataLoader(partial_train_ds, batch_size=4, shuffle=True)
valid_dataloader = torch.utils.data.DataLoader(partial_train_ds, batch_size=4, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# classes = train_dataloader.dataset.classes


class Net(Module):

    def __init__(self):
        super(Net, self).__init__()
#         COnv1
#         input shape = [32, 3, 128, 320]
        self.conv1 = Conv2d(3, 18, kernel_size=3, stride=1, padding=1)
#         outout shape [32, 18, 64, 160]
#         bathc normalization
        self.bn1 = nn.BatchNorm2d(num_features=18)
        self.relu1 = nn.ReLU()
        self.pool1 = MaxPool2d(kernel_size=2, stride=2, padding=0)
#         Conv2
#          output [32, 18, 64, 160] [4, 18, 64, 160]
        self.conv2 = Conv2d(18, 4, kernel_size=3, stride=1, padding=1)
        self.pool2 = MaxPool2d(kernel_size=2, stride=2, padding=0)
        
#         Conv3 [4, 4, 32, 80]
        self.conv3 = Conv2d(4, 4, kernel_size=3, stride=1, padding=1)
        self.pool3 = MaxPool2d(kernel_size=2, stride=2, padding=0)
        
#         output size [4, 4, 16, 40]
#     [4, 4, 32, 80]
        self.fc1 = Linear(4*16*40, 512)
        self.fc2 = Linear(512, 256)
        self.fc3 = Linear(256, 128)
        self.fc4 = Linear(128, 64)
        self.fc5 = Linear(64, 32)
        self.fc6 = Linear(32, 16)
        self.fc7 = Linear(16, 4)

    def forward(self, x):
#         conv1 layer
        x = F.relu(self.conv1(x))
        x= self.bn1(x)
        x = self.pool1(x)
        
#         conv2 layer
        x = F.relu(self.conv2(x))
#         x= self.bn2(x)
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
#         x= self.bn2(x)
        x = self.pool3(x)
#         print(x.size())
        x = x.view(-1, 4*16*40)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.fc7(x)
#         print(x.size())
        return x
        
        
net = Net().to(device)
import torch.optim as optim
import torch.nn as nn

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


train_loss_arr = []
valid_loss_arr = []
epochs = []
for epoch in range(150):  # loop over the dataset multiple times

    train_loss = 0.0
    net.train()
    for i, data in enumerate(train_dataloader):
#         print(len(data))
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
#         print(labels)
        
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        
#         # zero the parameter gradients
        optimizer.zero_grad()

#         # forward + backward + optimize
        outputs = net(inputs)
#         print(torch.isnan(outputs))
#         print(outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

#         # print statistics
        train_loss += loss.item()
        if i % 100 == 99:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, train_loss / 100))
            train_loss = 0.0
    net.eval()
    valid_loss = 0
    # turn off gradients for validation
    with torch.no_grad():
        for data, target in valid_dataloader:
            # forward pass
            data = data.to(device)
            target = target.to(device)
            output = net(data)
            # validation batch loss
            loss = criterion(output, target) 
            # accumulate the valid_loss
            valid_loss += loss.item()
    #########################
    ## PRINT EPOCH RESULTS ##
    #########################
    train_loss /= len(train_dataloader)
    valid_loss /= len(valid_dataloader)
    train_loss_arr.append(train_loss)
    valid_loss_arr.append(valid_loss)
    epochs.append(epoch+1)
    print(f'Epoch: {epoch+1}.. Training loss: {train_loss}.. Validation Loss: {valid_loss}')


print('Finished Training')

torch.save(net.state_dict(),'./cnn_models/model2.pth')


# train_loss_arr.append(train_loss)
# valid_loss_arr.append(valid_loss)
# epochs.append(epoch+1)
loss_csv = pd.DataFrame(np.array([epochs,train_loss_arr,valid_loss_arr]).T, columns=['epoch','train_loss','valid_loss'])
loss_csv.to_csv('./cnn_models/model_train_valid_loss1.csv',index=False)
