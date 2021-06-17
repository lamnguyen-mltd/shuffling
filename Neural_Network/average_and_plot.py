"""
Average data and plot
"""

import pandas as pd
from csv import reader
import numpy as np
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------
# Functions: average data and get the average data
def load_csv_result (filename):
    data = []
    with open (filename, 'r') as file:
        csv_reader = reader (file)
        for row in csv_reader:
            data.append(float(row[0]))
    return data[1:]

def avg_data (listrecord, liststats, result, record_path, record_avg_path, num_seed):
    # Get len(a)
    record_name = listrecord[0] + '_seed_0'
    a = np.array(load_csv_result (record_path + str(liststats[0]) + '_' + record_name + '.csv'))
    n = len(a)
    
    for record in listrecord:
        # Set the initial steps
        for stats in liststats:
            result[record][stats + '_avg'] = np.zeros(n)
            result[record][stats + '_std'] = np.zeros(n)

        # Get the avg data
        for seed in range (num_seed):
            record_seed = record + '_seed_' + str(seed)
            for stats in liststats:
                a = np.array(load_csv_result (record_path + str(stats) + '_' + record_seed + '.csv'))
                result[record][stats + '_avg'] += a / num_seed

        # Get the std data
        for seed in range (num_seed):
            record_seed = record + '_seed_' + str(seed)
            for stats in liststats:
                a = np.array(load_csv_result (record_path + str(stats) + '_' + record_seed + '.csv'))
                result[record][stats + '_std'] += (a - result[record][stats + '_avg'])**2 / num_seed # (Take sqrt later)

        # Save the average data for future references
        record_name = record + '_avg_' + str(num_seed) + '.csv'

        for stats in liststats:
            pd.DataFrame(result[record][stats + '_avg'] ).to_csv(record_avg_path + str(stats) + '_' + record_name, index = False)

        # Save the standard deviation data for future references
        record_name = record + '_std_' + str(num_seed) + '.csv'
        for stats in liststats:
            result[record][stats + '_std'] = np.sqrt(result[record][stats + '_std'])
            pd.DataFrame(result[record][stats + '_std'] ).to_csv(record_avg_path + str(stats) + '_' + record_name, index = False)

def get_avg_data (listrecord, liststats, result, record_avg_path, num_seed):
    for record in listrecord:
        # Get the data
        record_name = record + '_avg_' + str(num_seed) + '.csv'
        for stats in liststats:
            a = np.array(load_csv_result (record_avg_path + str(stats) + '_' + record_name))
            result[record][stats + '_avg'] = a

        # Get the STD data
        record_name = record + '_std_' + str(num_seed) + '.csv'
        for stats in liststats:
            a = np.array(load_csv_result (record_avg_path + str(stats) + '_' + record_name))
            result[record][stats + '_std'] = a

#-------------------------------------------------------------------------------
# Function plot data
def plot_data (dataname, num_epoch, listrecord, record_path, record_avg_path):
    liststats = ['loss', 'grad_loss', 'acc_train', 'acc_test']
    num_seed = 10

    #---------------------------------------------------------------------------
    # All the result will be keep in a dictionary:
    result = {}
    for name in listrecord: 
        result [name] = {}
    #---------------------------------------------------------------------------
    avg_data (listrecord, liststats, result, record_path, record_avg_path, num_seed)
    get_avg_data (listrecord, liststats, result, record_avg_path, num_seed)

    #---------------------------------------------------------------------------
    number = np.arange(num_epoch[1], num_epoch[0] + num_epoch[1], num_epoch[1])
    plt.figure()
    mark = ['^', 'v', '>', '<', 'o', 's', 'x', '>', 'd', 'o', 's', 'x', '>', 'd', 
            's', 'v', '>', '<', 'o', 's', 'x', '>', 'd', 'o', 's', 'x']
    plt.figure(figsize=(10,18))
    plt.title(dataname , fontsize = 28)

    # Plot loss
    plt.subplot(3,1,1)
    stats = 'loss'
    i = 0
    for record in listrecord:
        avg = result[record][stats + '_avg'][:]
        std = result[record][stats + '_std'][:]
        plt.plot (number[:], avg, marker = mark[i], label = listrecord[i].replace(dataname +'_','')) 
        plt.fill_between(number[:], avg-std, avg+std, alpha=0.3)
        i += 1
    plt.legend(fontsize = 18)         
    plt.xticks(size = 18)
    plt.yticks(size = 18)
    plt.xlabel('Number of effective passes', fontsize = 25)
    plt.ylabel('Train loss', fontsize = 25)
    plt.yscale('log')

    # Plot gradient
    plt.subplot(3,1,2)
    stats = 'grad_loss'
    i = 0
    for record in listrecord:
        avg = result[record][stats + '_avg'][:]
        std = result[record][stats + '_std'][:]
        plt.plot (number[:], avg, marker = mark[i], label = listrecord[i].replace(dataname +'_','')) 
        plt.fill_between(number[:], avg-std, avg+std, alpha=0.3)
        i += 1
    plt.legend(fontsize = 18)       
    plt.xticks(size = 18)
    plt.yticks(size = 18)
    plt.xlabel('Number of effective passes', fontsize = 25)
    plt.ylabel('Grad loss', fontsize = 25)
    plt.yscale('log')

    # Plot acc test
    plt.subplot(3,1,3)
    stats = 'acc_test'
    i = 0
    for record in listrecord:
        avg = result[record][stats + '_avg'][:]
        std = result[record][stats + '_std'][:]
        plt.plot (number[:], avg, marker = mark[i], label = listrecord[i].replace(dataname +'_','')) 
        plt.fill_between(number[:], avg-std, avg+std, alpha=0.3)
        i += 1
    plt.legend(fontsize = 18)       
    plt.xticks(size = 18)
    plt.yticks(size = 18)
    plt.xlabel('Number of effective passes', fontsize = 25)
    plt.ylabel('Test accuracy', fontsize = 25)
    plt.show()