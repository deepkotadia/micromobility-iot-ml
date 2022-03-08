from os import listdir, mkdir, makedirs
from os.path import isfile, join, isdir, split, splitext
import numpy as np
from preprocess_data import read_imu_stream_file, COL_NAMES
import pandas as pd
import shutil
import random

def split_data(dir_path):

    filenames = [join(dir_path, f) for f in listdir(dir_path) if isfile(join(dir_path, f))] 
    train_data = []
    val_data = []
    val_path = 'val'
    train_path = 'train'
    for file in filenames:
        indicator = random.random()

        if indicator < 0.8:
            train_data.append(file)
        else:
            val_data.append(file)
    if not isdir(join(dir_path, val_path)):
        mkdir(join(dir_path, val_path))
    if not isdir(join(dir_path, train_path)):
        mkdir(join(dir_path, train_path))

    for train_file in train_data:
        path_list = split(train_file)
        new_dir = join(path_list[0], train_path, path_list[-1])
        shutil.copy(src=train_file, dst=new_dir)
    for val_file in val_data:
        path_list = split(val_file)
        new_dir = join(path_list[0], val_path, path_list[-1])
        shutil.copy(src=val_file, dst=new_dir)

def make_data_samples(dir_path, stats_flag, window_size=256, step_size=64):
    print('starting sample making')
    filenames = [join(dir_path, f) for f in listdir(dir_path) if isfile(join(dir_path, f))] 
    print('found files: ', len(filenames))
    all_files = pd.DataFrame(columns=COL_NAMES)
    if not isdir(join(dir_path, 'samples')):
        makedirs(join(dir_path, 'samples'))
    for filename in filenames:
        data, _ = read_imu_stream_file(filename)
        all_files = all_files.append(data)
        
        for idx, i in enumerate(range(window_size, len(data), step_size)):
            sample = data.iloc[i-window_size:i]
            if 'sidewalk' in filename:
                label = 1
            else:
                label = 0
            sample['label'] = label
            file, ext = splitext(filename)
            save_dir_list = split(file)
            savename = f'{save_dir_list[0]}/samples/{save_dir_list[1]}_{idx}{ext}'
            sample.to_csv(savename)
    print('sample making done, computing metrics...')    
    all_files = all_files[['accl_x', 'accl_y', 'accl_z', 'gyro_x', 'gyro_y', 'gyro_z']]
    median = all_files.median(axis=0, skipna=True).to_numpy()
    mean = all_files.mean(axis=0, skipna=True).to_numpy()
    std = all_files.std(axis=0, skipna=True).to_numpy()
    low = all_files.quantile(q=0.1, axis=0).to_numpy()
    high = all_files.quantile(q=0.9, axis=0).to_numpy()
    intQrange = high - low
    stats = np.vstack([median, intQrange, high, low, mean, std])
    stats = pd.DataFrame(stats, columns=['median', 'intQrange', 'high', 'low', 'mean', 'std'], index=['accl_x', 'accl_y', 'accl_z', 'gyro_x', 'gyro_y', 'gyro_z'])
    #stats = pd.concat([median, intQrange], axis=1).reindex(median.index)
    #stats = pd.concat((stats, low), axis=1)
    #stats = pd.DataFrame(stats, columns=['median', 'range','low', 'high', 'mean', 'std'], index=all_files.columns)
    #stats.
    stats.to_csv(f'IMU_Data/data_stats_{stats_flag}.csv')
    print('saved metrics file')

if __name__=='__main__':
    dir_path = 'IMU_Data/train'
    make_data_samples(dir_path, stats_flag='train')
    val_path = 'IMU_Data/val'
    make_data_samples(val_path, stats_flag='val')
    #split_data(dir_path)
    
