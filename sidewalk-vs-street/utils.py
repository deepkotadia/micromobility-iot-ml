import numpy as np
from numpy.lib.shape_base import _make_along_axis_idx
from numpy.lib.type_check import nan_to_num
import pandas as df
import matplotlib.pyplot as plt
import glob
import os

def visualize_imu(df, title):
    '''plots raw accelerometer of one data sample in 3 axes'''
    imu_x1 = df.loc[:1000,['acc_x']]
    imu_y1 = df.loc[:1000,['acc_y']]
    imu_z1 = df.loc[:1000,['acc_z']]

    fig, ax = plt.subplots(3, 1, sharex='all')
    fig.add_subplot(111, frameon=False)
    plt.ylabel("acceleration (m/s**2)", labelpad=20, fontweight='bold')
    plt.xlabel("time", fontweight='bold')
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    ax[0].plot(imu_x1, color='red', marker='o')
    ax[0].set_title("x axis", fontsize=15)
    ax[0].set_ylim([-5, 5])

    
    ax[1].plot(imu_y1, color='blue', marker='o')
    ax[1].set_title("y axis", fontsize=15)
    ax[1].set_ylim([4, 15])

    ax[2].plot(imu_z1, color='green', marker='o')
    ax[2].set_title("z axis", fontsize=15)
    ax[2].set_ylim([-7, 4])
    fig.suptitle(title)
    plt.savefig("plots/"+title+".png")
    plt.close()

    

def compute_features(datastream, title):
    '''compute features for each datapoint:
        input: one sample (pandas df), title for filename (str)
        output: sample dataframe w features added in columns'''
    mean_x = np.mean(datastream['acc_x'])
    mean_y = np.mean(datastream['acc_y'])
    mean_z = np.mean(datastream['acc_z'])
    datastream['acc_std_x'] = (datastream["acc_x"]-mean_x)**2
    datastream['acc_std_y'] = (datastream["acc_y"]-mean_y)**2
    datastream['acc_std_z'] = (datastream["acc_z"]-mean_z)**2
    datastream.to_csv("IMU_Streams/Features/"+title+".csv")
    return datastream

def plot_std(datastream, title):
    '''plots the standard deviation of one sample in 3 axes'''
    fig, ax = plt.subplots(3, 1, sharex='all')
    fig.add_subplot(111, frameon=False)
    plt.ylabel("deviation from mean acceleration (m/s**2)**2", labelpad=20, fontweight='bold')
    plt.xlabel("time", fontweight='bold')
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    ax[0].plot(datastream.loc[:500, ['acc_std_x']], color='red', marker='o')
    ax[0].set_title("x axis", fontsize=15)
    ax[0].set_ylim([0,30])

    
    ax[1].plot(datastream.loc[:500, ['acc_std_y']], color='blue', marker='o')
    ax[1].set_title("y axis", fontsize=15)
    ax[1].set_ylim([0,30])


    ax[2].plot(datastream.loc[:500, ['acc_std_z']], color='green', marker='o')
    ax[2].set_title("z axis", fontsize=15)
    ax[2].set_ylim([0,30])

    fig.suptitle(title)
    plt.show()
    plt.savefig("plots/"+title+"_std.png")
    plt.close()
    

def plot_all():
    '''runs functions for all files'''
    files = glob.glob("IMU_Streams/*.csv")
    for path in files:
        data = df.read_csv(path, dtype=float, names=[0, 1, "acc_x", "acc_y", "acc_z", 5, "gyr_x", 'gyr_y', 'gyr_z', 9, "magnet_x", "magnet_y", 'magnet_z'])
        title, ext = os.path.splitext(path)
        _, title = os.path.split(title)
        data_std = compute_features(data, title)
        visualize_imu(data_std, title)
        plot_std(data_std, title)
        

plot_all()

'''note: in CSV files, x, y, z directions are columns 2, 3, 4'''