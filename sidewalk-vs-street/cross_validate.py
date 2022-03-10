from cnn_model import *
from dataloader import SidewalkDataSet
from os import listdir, makedirs, mkdir
from os.path import isfile, join, isdir, split, splitext
import numpy as np
from matplotlib.pyplot import step
from preprocess_data import read_imu_stream_file, COL_NAMES
import pandas as pd
import shutil
import random
import numpy as np
from collections import defaultdict

def split_data(dir_path, idx):
    print('splitting train/val')
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
    file_name = f'train_test_files{idx}.txt'
    with open(file_name, 'w') as f:
        f.write('train data: \n')
        f.writelines(train_data)
        f.write('\n Val Data: \n')
        f.writelines(val_data)
    return train_data, val_data
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

def make_data_sample(main_path, data_paths, fold, split_idx, window_size=256, step_size=1):
    print(f'making {fold} samples')
    #filenames = [join(dir_path, f) for f in listdir(dir_path) if isfile(join(dir_path, f))] 
    all_files = pd.DataFrame(columns=COL_NAMES)
    
    new_path = f'{main_path}/split_{split_idx}/{fold}'
    if not isdir(new_path):
        makedirs(new_path)

    for filename in data_paths:
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
            temp = f'{main_path}/split_{split_idx}/{fold}'
            savename = f'{new_path}/{save_dir_list[1]}_{idx}{ext}'
            #savename = f'{save_dir_list[0]}/samples/{save_dir_list[1]}_{idx}{ext}'
            sample.to_csv(savename)
        
    all_files = all_files[['accl_x', 'accl_y', 'accl_z', 'gyro_x', 'gyro_y', 'gyro_z']]
    median = all_files.median(axis=0, skipna=True).to_numpy()
    mean = all_files.mean(axis=0, skipna=True).to_numpy()
    std = all_files.std(axis=0, skipna=True).to_numpy()
    low = all_files.quantile(q=0.1, axis=0).to_numpy()
    high = all_files.quantile(q=0.9, axis=0).to_numpy()
    intQrange = high - low
    stats = np.vstack([median, intQrange, high, low, mean, std])
    stats = pd.DataFrame(stats, columns=['median', 'intQrange', 'high', 'low', 'mean', 'std'], index=['accl_x', 'accl_y', 'accl_z', 'gyro_x', 'gyro_y', 'gyro_z'])
    const_path = f'{main_path}/split_{split_idx}'
    constants_path = f'{const_path}/data_stats_{fold}_{split_idx}.csv'
    stats.to_csv(constants_path)
    print('done here')
    return new_path, constants_path

def cross_validate(dir_path):

    runs = 5 
    

    results_dict = defaultdict(list)
    res_file =  open('cross_val_results_file_test.txt', 'a')
    for i in range(runs):
        res_file.write(f"RUN {i} \n")
        print(f"------------RUN: {i}----------------")
        try_train_path = f'IMU_Data/split_{i}/train'
        try_val_path = f'IMU_Data/split_{i}/val'
        if not (isdir(try_train_path) and isdir(try_val_path)):
            train_data, val_data = split_data(dir_path, i)
            print('making data')
            train_path, constants_path = make_data_sample(dir_path, train_data, fold='train', split_idx=i)
            val_path, _ = make_data_sample(dir_path, val_data, fold='val', split_idx=i)
            print('done with data, start training model')
        else:
            print('data exists, start training now')
            train_path = try_train_path
            val_path = try_val_path
            constants_path = f'{dir_path}/split_{i}/data_stats_train_{i}.csv'
        trainer = run_trainer(train_path, val_path, constants_path, split_idx=i)
        accuracy, f1, precision, recall = validate(trainer)
        results_dict['f1'].append(f1)
        results_dict['accuracy'].append(accuracy)
        results_dict['precision'].append(precision)
        results_dict['recall'].append(recall)
        #results_dict['conf_matrix'].append()
        print(results_dict)
        res_file.write(str(dict(results_dict)))
        res_file.write('\n')
    print(f"avg metrics over {runs} folds")
    for metric in ['f1', 'accuracy', 'precision', 'recall']:
        avg = torch.mean(torch.tensor(results_dict[metric]))
        print(f"Metric: {metric}, Average: {avg}")
        res_file.write(f"Metric: {metric}, Average: {avg}")
    res_file.close()


if __name__ == "__main__":
    dir_path = 'IMU_Data'
    cross_validate(dir_path="IMU_Data")

