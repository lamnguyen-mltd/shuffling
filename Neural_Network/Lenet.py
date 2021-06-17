"""
Define Neural Networks 
"""

import torch.nn as nn
import torch.nn.functional as F

#---------------(LeNet_300_100 for Cifar10)-------------------------------------
class LeNet_300_100_cifar (nn.Module):
    def __init__ (self):
        super().__init__()

        # This network is for images of size 32x32, with 3 color channels
        self.fc1 = nn.Linear (32 *32 *3, 300)
        self.fc2 = nn.Linear (300, 100)
        self.fc3 = nn.Linear (100, 10)
        self.relu = nn.ReLU()
        self.lastbias = 'fc3.bias'
    
    def forward (self, x):
        x = x.view(-1, 32 * 32 * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
#-------------------------------------------------------------------------------


#---------------(LeNet_300_100 for (Fashion) MNIST)-----------------------------
class LeNet_300_100 (nn.Module):
    def __init__ (self):
        super().__init__()

        # This network is for images of size 28x28, with 1 color channels 
        self.fc1 = nn.Linear (28* 28, 300)
        self.fc2 = nn.Linear (300, 100)
        self.fc3 = nn.Linear (100, 10)
        self.relu = nn.ReLU()
        self.lastbias = 'fc3.bias'
    
    def forward (self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
#-------------------------------------------------------------------------------