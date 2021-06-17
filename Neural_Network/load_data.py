"""
Load Data 

dataname: 'mnist' or 'cifar10' 
"""

import torch
import torchvision 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset

import tensorflow as tf
import keras
from keras.datasets import fashion_mnist, mnist, cifar10, cifar100
import numpy as np


def load_data (dataname, batch, shuffle='RR'):
    print('--- Loading data', dataname, 'with batch size', batch)

    if (dataname == 'cifar10'):
        # input image dimensions
        img_rows, img_cols, num_layers = 32, 32, 3

        # the data, shuffled and split between train and test sets
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    elif (dataname == 'mnist'):
        # input image dimensions
        img_rows, img_cols, num_layers = 28, 28, 1

        # the data, shuffled and split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()  


    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, num_layers)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, num_layers)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_train = y_train.reshape(y_train.shape[0])
    y_test = y_test.reshape(y_test.shape[0])

    x_mean = np.mean(x_train, axis = (0,1,2))
    x_std = np.std(x_train, axis = (0,1,2))
    x_train = (x_train - x_mean) / x_std
    x_test = (x_test - x_mean) / x_std

    if shuffle == 'SO': # (Shuffle Once at the beginning)
        num_samples = x_train.shape[0]
        index_array = np.array([i for i in range(num_samples)])
        np.random.shuffle(index_array)
        x_train = x_train[index_array]
        y_train = y_train[index_array]

    x_tensor = torch.from_numpy(x_train).float()
    y_tensor = torch.from_numpy(y_train).long()
    x_tensor_test = torch.from_numpy(x_test).float()
    y_tensor_test = torch.from_numpy(y_test).long()
    train_set = TensorDataset(x_tensor, y_tensor)
    test_set = TensorDataset(x_tensor_test, y_tensor_test)

    if shuffle == 'RR':
        train_loader = DataLoader(train_set, batch_size=batch, shuffle=True) # (Reshuffle every epoch)
    if shuffle == 'IG' or shuffle == 'SO':
        train_loader = DataLoader(train_set, batch_size=batch, shuffle=False) # (NO shuffle every epoch)

    test_loader = DataLoader(test_set, batch_size=batch, shuffle=False)

    return train_loader, test_loader