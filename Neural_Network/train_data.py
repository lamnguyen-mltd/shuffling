""" 
Training
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import pandas as pd

from load_data import *
from algorithms import *
from record_history import *
from Lenet import *
from schedule_LR import *

def train_data (dataname, num_epoch, namelr, params, listrecord, record_path):
    # Get params
    listgamma, listalpha, listshuffle = params

    #-------------------------------------------------------------------------------
    # Start training

    for gamma in listgamma:
      for alpha in listalpha:
        for shuffle in listshuffle:
          for seed in range(10):
              # Record name: 
              alg = '_SGD_' + shuffle + '_' + namelr

              print('Step 0: Load data and start network')
              # Load data
              train_loader, test_loader = load_data (dataname, 256, shuffle)

              # Start network
              if (dataname == 'cifar10'):
                net = LeNet_300_100_cifar()
                name_net = LeNet_300_100_cifar 

              if (dataname == 'mnist'):
                net = LeNet_300_100()
                name_net = LeNet_300_100
              print('---', name_net, 'started')

              # GPU :)
              device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
              net.to(device)
              criterion = nn.CrossEntropyLoss()

              # Pick LR scheme: 
              if namelr == 'const_':
                scheduleLR = constant (gamma)
                record_name = dataname + alg + str(gamma) + '_seed_' + str(seed)
              elif namelr == 'dim_':
                scheduleLR = diminishing (gamma, alpha)
                record_name = dataname + alg + str(gamma) + '_' + str(alpha) + '_seed_' + str(seed)
            
              # Start training
              SGD_train (name_net, num_epoch, train_loader, test_loader, scheduleLR, criterion, record_path, record_name)
    return listrecord