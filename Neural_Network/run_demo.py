"""
Run the experiments
"""

import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import pandas as pd
import random

from load_data import *
from algorithms import *
from record_history import *
from Lenet import *
from schedule_LR import *
from train_data import *
from average_and_plot import *

# Change the record path 
record_path = './Record/'
record_avg_path = record_path + 'Avg/'

if not os.path.exists(record_path):
    os.makedirs(record_path)

if not os.path.exists(record_avg_path):
    os.makedirs(record_avg_path)


# Experiment 2: Comparing the diminishing LRs for neural networks --------------

namelr = 'dim_'
num_epoch = [40, 4] # Run for 40 epochs, and measure the performance each 4 epochs

# Data: Mnist ------------------------------------------------------------------
dataname = 'mnist' 
listrecord = []

listgamma = [0.05, 0.1]
listalpha = [1/3, 1/2]
listshuffle = ['RR']
params = listgamma, listalpha, listshuffle
listrecord = train_data (dataname, num_epoch, namelr, params, listrecord, record_path)


plot_data (dataname, num_epoch, listrecord, record_path, record_avg_path)


# Experiment 3: Comparing different shuffling schemes for Cifar10 --------------
namelr = 'const_'
num_epoch = [40, 4] # Run for 40 epochs, and measure the performance each 4 epochs

# Data: Cifar10 ----------------------------------------------------------------
dataname = 'cifar10' 
listrecord = []

listgamma = [0.01]
listalpha = [0]
listshuffle = ['RR', 'IG', 'SO']
params = listgamma, listalpha, listshuffle
listrecord = train_data (dataname, num_epoch, namelr, params, listrecord, record_path)