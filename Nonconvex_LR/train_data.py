""" 
Training
"""

import numpy as np
import pandas as pd

from load_data import *
from algorithms import *
from record_history import *
from util_func import *
from schedule_LR import *

def train_data (dataname, num_epoch, namelr, params, listrecord, record_path):
    # Loading data and start network
    listgamma, listalpha = params
    #-------------------------------------------------------------------------------
    print('Step 0: Import data')
    # Load data
    X, Y, X_test, Y_test = import_data(dataname)
    lamb =  0.01
    shuffle = 'RR'

    #-------------------------------------------------------------------------------
    # Start training

    for gamma in listgamma:
      for alpha in listalpha:

          # Record name: 
          alg = '_SGD_' + namelr 

          # Pick LR scheme: 
          if namelr == 'const_':
            scheduleLR = constant (gamma)
            record_name = dataname + alg + str(gamma) 
          elif namelr == 'dim_':
            scheduleLR = diminishing (gamma, alpha)
            record_name = dataname + alg + str(gamma) + '_' + str(alpha) 

          listrecord.append(record_name)
          for seed in range(10):
            record_seed = record_name + '_seed_' + str(seed)
            SGD_train (X, Y, X_test, Y_test, num_epoch, shuffle, lamb, scheduleLR, record_path, record_seed)
    return listrecord