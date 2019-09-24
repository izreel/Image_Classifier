import torch.nn as nn
import torch.nn.functional as F

class ObjectNet(nn.Module):
    def __init__(self):
        super(ObjectNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding= 1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding= 1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding= 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p= 0.2)
        self.fc1 = nn.Linear(4*4*64, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        #Flattens image in order to go through the fully connnected layers
        x = x.view(-1, 4*4*64)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(self.fc3(x))

        return x

