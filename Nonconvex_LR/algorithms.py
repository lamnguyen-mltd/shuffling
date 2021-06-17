"""
Algorithms
"""

# Stochastic Gradient Method

import numpy as np
import random
import pandas as pd
from util_func import *
from record_history import *
from schedule_LR import *


#-------------------------------------------------------------------------------
def SGD_train (X, Y, X_test, Y_test, num_epoch, shuffle = 'RR', lamb = None, scheduleLR = None, record_path = None, record_name = None):

    #---------------------------------------------------------------------------
    print('Step 1: Start training with learning rate', scheduleLR(1)) 
    num_train, num_feature = np.shape(X)
    num_test = len(Y_test)
    w = 0.5*np.ones(num_feature)
    # Initialization step to record the results
    record_history ('initial', 0, X, Y, X_test, Y_test, w, lamb, record_path, record_name)

    #---------------------------------------------------------------------------
    print('Step 2: Training', num_epoch[0], 'epoch(s) for', record_name)
    for epoch in range (num_epoch[0]):
        lr = scheduleLR(epoch + 1)

        # Choosing shuffling or not:
        if shuffle == 'RR':
            index_set = [i for i in range(0, num_train)]
            random.shuffle(index_set)
        else:
            index_set = np.random.randint(num_train, size=num_train)

        # Training
        for j in range(0, num_train):
            # Evaluate component gradient
            grad_i = grad_com_val(index_set[j], num_train, num_feature, X, Y, w, lamb)
            
            # Algorithm update
            w = w - lr * grad_i

        # Measure and add statistics after a number of epoch --------------------
        if (epoch+1) % (num_epoch[1]) == 0:
            record_history ('measure', epoch + 1, X, Y, X_test, Y_test, w, lamb, record_path, record_name)
    #---------------------------------------------------------------------------
    print('Step 3: Save the results') 
    record_history ('save', 0, X, Y, X_test, Y_test, w, lamb, record_path, record_name)
    print('Finish Training \n ------------------------------------------------')

