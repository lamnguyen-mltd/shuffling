"""
Run the experiments
"""

import os
import numpy as np
import pandas as pd

from load_data import *
from algorithms import *
from record_history import *
from util_func import *
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

# Experiment 1: Comparing the diminishing LRs for logistic regression ----------
namelr = 'dim_'
num_epoch = [10, 2] # Run for 10 epochs, and measure the performance each 2 epochs

# Data: w8a --------------------------------------------------------------------
dataname = 'w8a'
listrecord = []

listgamma = [0.001, 0.005]
listalpha = [1/3, 1/2]
params = listgamma, listalpha
listrecord = train_data (dataname, num_epoch, namelr, params, listrecord, record_path)
plot_data (dataname, num_epoch, listrecord, record_path, record_avg_path)


