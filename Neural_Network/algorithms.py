"""
Algorithms
"""

import torch 
import pandas as pd
from record_history import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import random

#-------------------------------------------------------------------------------
def SGD_train (NameNet, num_epoch, loader, test_loader, scheduleLR, criterion, record_path, record_name):

    #---------------------------------------------------------------------------
    print('Step 1: Restart network with starting learning rate', scheduleLR(1))
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    net = NameNet() 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device) 
    
    # Initialize to save data
    record_history ('initial', 0, net, loader, test_loader, criterion, record_path, record_name)
    seed_alg = random.randint(0,100)
    torch.manual_seed(seed_alg)

    #---------------------------------------------------------------------------
    print('Step 2: Train the network', num_epoch[0], 'epoch(s) for', record_name)
    
    for epoch in range (num_epoch[0]):
        lr = scheduleLR(epoch + 1)

        # Start training for an epoch ------------------------------------------        
        for i, data in enumerate (loader, 0):
            # Get the inputs and labels ----------------------------------------
            inputs, labels = data[0].to(device), data[1].to(device)
            
            # Zero grad all the trainable parameters ---------------------------
            net.zero_grad() 

            # Forward pass and backward pass -----------------------------------
            outputs = net(inputs)
            loss = criterion (outputs, labels)
            loss.backward() 
            
            # Loop through the weights -----------------------------------------
            for name, f in net.named_parameters():
                # Update rule
                f.data.add_(f.grad.data, alpha = -lr)

        # Measure and add statistics after a number of epoch --------------------
        if (epoch+1) % (num_epoch[1]) == 0:
            record_history ('measure', epoch + 1, net, loader, test_loader, criterion, record_path, record_name)
    #---------------------------------------------------------------------------
    print('Step 3: Save the results') 
    record_history ('save', 0, net, loader, test_loader, criterion, record_path, record_name)
    print('Finish Training \n ------------------------------------------------')

