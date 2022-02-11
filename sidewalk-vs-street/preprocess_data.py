from cProfile import run
from os import listdir
from os.path import isfile, join
import numpy as np
from numpy.lib.function_base import percentile
import pandas as pd
from scipy.sparse import data
from sklearn import feature_selection
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter, find_peaks, lfilter
from sklearn.utils import shuffle
import random


COL_NAMES =['time',
           'accl_id', 'accl_x', 'accl_y', 'accl_z',
           'gyro_id', 'gyro_x', 'gyro_y', 'gyro_z',
           'mag_id', 'mag_x', 'mag_y', 'mag_z']

FEATURE_COLS = ['mean_accl_x', 'mean_accl_y', 'mean_accl_z', 'mean_gyro_x', 'mean_gyro_y', 
    'mean_gyro_z', 'std_accl_x', 'std_accl_y', 'std_accl_z', 'std_gyro_x', 'std_gyro_y', 'std_gyro_z', 
    '90th_accl_x', '90th_accl_y', '90th_accl_z', '90th_gyro_x', '90th_gyro_y', '90th_gyro_z', 
    '10th_accl_x', '10th_accl_y', '10th_accl_z', '10th_gyro_x', '10th_gyro_y', '10th_gyro_z', 
    'range_accl_x', 'range_accl_y', 'range_accl_z', 'range_gyro_x', 'range_gyro_y', 'range_gyro_z', 
    'peaks_accl_x', 'peaks_accl_y', 'peaks_accl_z', 'peaks_gyro_x', 'peaks_gyro_y', 'peaks_gyro_z', 
    'med_prominence_a_x', 'med_prominence_a_y', 'med_prominence_a_z', 
    'med_prominence_g_x', 'med_prominence_g_y', 'med_prominence_g_z']


def read_imu_stream_file(filepath):
    df = pd.read_csv(filepath_or_buffer=filepath, names=COL_NAMES, dtype=float)
    full_data_stream = df[['accl_x', 'accl_y', 'accl_z', 'gyro_x', 'gyro_y', 'gyro_z']]
    # gyro_df = df[['gyro_x', 'gyro_y', 'gyro_z']]
    # mag_df = df[['mag_x', 'mag_y', 'mag_z']]
    full_data_stream = full_data_stream[150:-150]
    return full_data_stream, full_data_stream.shape[0]


def quantize_and_clean(sensor_reading_triplets, num_triplets_per_window, col_labels, feature_label):
    filtered_triplets = sensor_reading_triplets[150:-150]  # discard first and last 3s of data (sampling rate is 50Hz)

    # Quantize by window size
    flattened_triplets = filtered_triplets.flatten()
    slice_idx = flattened_triplets.shape[0] - (flattened_triplets.shape[0] % (3 * num_triplets_per_window))
    sliced_flattened_triplets = flattened_triplets[:slice_idx]
    quantized = sliced_flattened_triplets.reshape(-1, 3 * num_triplets_per_window)

    # Add label col (sidewalk or street
    quantized_df = pd.DataFrame(quantized, columns=col_labels)
    quantized_df['label'] = feature_label
    return quantized_df


def normalize(df, mean, std):
    """
    z normalization (mean 0, std 1)
    """
    numerator = (df-mean)
    normalized_df = numerator / std
    return normalized_df


def read_all_stream_files_in_dir(dir_path, test_size=0.15, time_window=10, mode='fixed', shuffle=True):
    print("parameters for data preprocessing: ")
    print("mode: ", mode)
    print("Shuffle: ", shuffle)
    print(f"window size: {time_window} s")
    """
    reads all data streams as csv, normalize, divide into training samples and label
    prints data stats and returns dataframe of all training samples
    """
    filenames = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]  # get all stream filenames

    col_names = list()
    for i in range(time_window):
        col_names.extend([f'accl_x_{i}', f'accl_y_{i}', f'accl_z_{i}'])

    constants  = pd.read_csv('normalizing_constants.csv', dtype=float)
    mean_ = constants.loc[0].to_numpy()
    std_ = constants.loc[1].to_numpy()
    # Different types of streets and sidewalks (for secondary labels)
    df_sidewalk1 = pd.DataFrame()
    df_sidewalk = pd.DataFrame()
    df_street1 = pd.DataFrame()
    df_street2 = pd.DataFrame()
    df_street3 = pd.DataFrame()
    train = pd.DataFrame()
    test = pd.DataFrame()
    train_files = open('train_files.txt', 'w')
    test_files = open('test_files.txt', 'w')
    for filename in filenames:
        data_df, num_rows = read_imu_stream_file(f'{dir_path}/{filename}')
        #print('data types:',  data_df.dtypes)
        #cols = data_df.columns
        #data_df[cols] = data_df[cols].apply(pd.to_numeric)
        #print("dtypes after convert: ", data_df.dtypes)
        data_df[['gyro_x', 'gyro_y', 'gyro_z']] = data_df[['gyro_x', 'gyro_y', 'gyro_z']].fillna(value=data_df[['gyro_x', 'gyro_y', 'gyro_z']].mean())
        #data_df = pd.DataFrame(filter_data(data_df), columns=['accl_x', 'accl_y', 'accl_z', 'gyro_x', 'gyro_y', 'gyro_z'])
        data_df = pd.DataFrame(low_pass_filter(data_df, beta=1, alpha=0.96), columns=['accl_x', 'accl_y', 'accl_z', 'gyro_x', 'gyro_y', 'gyro_z'])
        if mode == 'running_window':
            data_stream = running_window(data_df, time_window=time_window)
        elif mode == 'fixed':
            data_stream = samples_and_feature_extraction(data_df, time_window=time_window)
        indicator = random.random()

        # append to corresponding df and label
        if 'sidewalk1' in filename:
            
            data_stream['label'] = 1
            data_stream['sublabel'] = 'sidewalk1'
            if df_sidewalk1.empty:
                df_sidewalk1 = data_stream
            else: 
                df_sidewalk1 = df_sidewalk1.append(data_stream, ignore_index=True)
            if indicator < 0.8:
                train = train.append(data_stream)
                train_files.write(filename + '\n')
            else:
                test = test.append(data_stream)
                test_files.write(filename + '\n')
        elif 'sidewalk' in filename:
            indicator = random.random()
            data_stream['label'] = 1
            data_stream['sublabel'] = 'sidewalk'
            if df_sidewalk.empty:
                df_sidewalk = data_stream
            else:
                df_sidewalk = df_sidewalk.append(data_stream, ignore_index=True)
            if indicator < 0.8:
                train = train.append(data_stream)
                train_files.write(filename + '\n')
            else:
                test = test.append(data_stream)
                test_files.write(filename + '\n')
        elif 'street1' in filename or 'st1' in filename:
            
            data_stream['label'] = 0
            data_stream['sublabel'] = 'street1'
            if df_street1.empty:
                df_street1 = data_stream
            else:
                df_street1 = df_street1.append(data_stream, ignore_index=True)
            if indicator < 0.8:
                train = train.append(data_stream)
                train_files.write(filename + '\n')
            else:
                test = test.append(data_stream)
                test_files.write(filename + '\n')
        elif 'street2' in filename or 'st2' in filename:
            
            data_stream['label'] = 0
            data_stream['sublabel'] = 'street2'
            if df_street2.empty:
                df_street2 = data_stream
            else:
                df_street2 = df_street2.append(data_stream, ignore_index=True)
            if indicator < 0.8:
                train = train.append(data_stream)
                train_files.write(filename + '\n')
            else:
                test = test.append(data_stream)
                test_files.write(filename + '\n')
        elif 'street3' in filename or 'st3' in filename:
            
            data_stream['label'] = 0
            data_stream['sublabel'] = 'street3'
            if df_street3.empty:
                df_street3 = data_stream
            else:
                df_street3 = df_street3.append(data_stream, ignore_index=True)
            if indicator < 0.8:
                train = train.append(data_stream)
                train_files.write(filename + '\n')
            else:
                test = test.append(data_stream)
                train_files.write(filename + '\n')

    full_df = pd.concat((df_sidewalk, df_sidewalk1, df_street1, df_street2, df_street3), axis=0)
    mean = full_df[FEATURE_COLS].mean(axis=0)
    std = full_df[FEATURE_COLS].std(axis=0)
    if shuffle:
        train = train.sample(frac=1).reset_index(drop=True)
        test = test.sample(frac=1).reset_index(drop=True)

    trainlabel = train['label']
    testlabel = test['label']
    train_sublabel = train['sublabel']
    test_sublabel = test['sublabel']
    train = train.drop(columns=['label', 'sublabel'])
    test = test.drop(columns=['label', 'sublabel'])
    train = normalize(train, mean, std)
    test = normalize(test, mean, std)
    train['label'] = trainlabel
    test['label'] = testlabel
    train['sublabel'] = train_sublabel
    test['sublabel'] = test_sublabel



    '''

    #try train and test on entirely different sidewalk/street:
    
    train = pd.concat((normalized_sidewalk, normalized_street1, normalized_street2))
    test = pd.concat((normalized_sidewalk1, normalized_street3))
    if shuffle:
        train.sample(frac=1).reset_index(drop=True)
        test.sample(frac=1).reset_index(drop=True)
    print('sublabels in train', pd.unique(train['sublabel']))
    print('sublabels in test: ', pd.unique(test['sublabel']))'''
    
    print(f"number of samples: train: {train.shape[0]} test: {test.shape[0]}")

    return train, test


def shuffle_and_split(df, test_size=0.2, shuffle=True):
    train, test = train_test_split(df, test_size=test_size, shuffle=shuffle)
    return train, test


def samples_and_feature_extraction(dataframe, time_window=10, filter=None):
    window_size = 50*time_window
    '''divides into training points of size window_size (default 150 samples = 3 seconds)
        computes features (mean, std, percentiles)
        returns dataframe of dim (num_samples,num_features)'''
    col_names = ['mean_accl_x', 'mean_accl_y', 'mean_accl_z', 'mean_gyro_x', 'mean_gyro_y', 
    'mean_gyro_z', 'std_accl_x', 'std_accl_y', 'std_accl_z', 'std_gyro_x', 'std_gyro_y', 'std_gyro_z', 
    '90th_accl_x', '90th_accl_y', '90th_accl_z', '90th_gyro_x', '90th_gyro_y', '90th_gyro_z', 
    '10th_accl_x', '10th_accl_y', '10th_accl_z', '10th_gyro_x', '10th_gyro_y', '10th_gyro_z', 
    'range_accl_x', 'range_accl_y', 'range_accl_z', 'range_gyro_x', 'range_gyro_y', 'range_gyro_z']
    idx = dataframe.shape[0] - (dataframe.shape[0] % window_size)
    dataframe = dataframe[:idx]
    num_samples = dataframe.shape[0]/window_size
    np_df = dataframe.to_numpy()
    splits = np.array(np.vsplit(np_df, num_samples))
    mean = np.nanmean(splits, axis=1)
    std = np.nanstd(splits, axis=1)
    percentile_90th = np.nanpercentile(splits, q=90, axis=1)
    percentile_10th = np.nanpercentile(splits, q=10, axis=1)
    spread = percentile_90th - percentile_10th  
    samples = np.hstack((mean, std, percentile_90th, percentile_10th, spread))
    all_samples = pd.DataFrame(samples, columns=col_names)
    if filter == 'savgol':
        num_peaks = np.zeros((len(splits)))
        for i, split in enumerate(splits):
            filtered = savgol_filter(split, window_length=5, polyorder=3, axis=1)
            peaks = find_peaks(filtered)
            num_peaks[i] = len(peaks)
        all_samples['num_peaks'] = num_peaks
    return all_samples

def running_window(dataframe, time_window=10):
    window_size = time_window*50
    col_names = ['mean_accl_x', 'mean_accl_y', 'mean_accl_z', 'mean_gyro_x', 'mean_gyro_y', 
    'mean_gyro_z', 'std_accl_x', 'std_accl_y', 'std_accl_z', 'std_gyro_x', 'std_gyro_y', 'std_gyro_z', 
    '90th_accl_x', '90th_accl_y', '90th_accl_z', '90th_gyro_x', '90th_gyro_y', '90th_gyro_z', 
    '10th_accl_x', '10th_accl_y', '10th_accl_z', '10th_gyro_x', '10th_gyro_y', '10th_gyro_z', 
    'range_accl_x', 'range_accl_y', 'range_accl_z', 'range_gyro_x', 'range_gyro_y', 'range_gyro_z', 
    'peaks_accl_x', 'peaks_accl_y', 'peaks_accl_z', 'peaks_gyro_x', 'peaks_gyro_y', 'peaks_gyro_z', 
    'med_prominence_a_x', 'med_prominence_a_y', 'med_prominence_a_z', 
    'med_prominence_g_x', 'med_prominence_g_y', 'med_prominence_g_z']

    all_samples = np.zeros((dataframe.shape[0], len(col_names)))
    all_samples = pd.DataFrame(all_samples, columns=col_names)
    for i in range(window_size, len(dataframe)):
        slice = dataframe.iloc[i-window_size:i,:]
        mean = np.nanmean(slice, axis=0)
        std = np.nanstd(slice, axis=0)
        percentile_90th = np.nanpercentile(slice, q=0.9, axis=0)
        percentile_10th = np.nanpercentile(slice, q=0.1, axis=0)
        spread = percentile_90th - percentile_10th
        num_peaks, med_prominence = peak_finder(slice)
        row = np.hstack((mean, std, percentile_90th, percentile_10th, spread, num_peaks, med_prominence))
        all_samples.iloc[i,:] = row
    all_samples = all_samples.iloc[window_size:,:]
    return all_samples

def filter_data(data):
    filtered = savgol_filter(data, window_length=75, polyorder=3, axis=0)
    return filtered

def low_pass_filter(data, beta, alpha):
    return lfilter(b=[beta], a=[alpha], x=data, axis=0)

def peak_finder(signal):
    peaks_list = np.zeros((signal.shape[1]))
    prominences = np.zeros((signal.shape[1]))
    cols = signal.columns
    for idx, col in enumerate(signal.columns):
        peaks, properties = find_peaks(signal[col], prominence=0.03)
        num_peaks = len(peaks)
        if num_peaks > 0:
            med_prominence = np.median(properties['prominences'])
        else:
            med_prominence = 0
        peaks_list[idx] = num_peaks
        prominences[idx] = med_prominence
    return peaks_list, prominences



if __name__ == '__main__':
    time_window = 10
    train, test = read_all_stream_files_in_dir('IMU_Streams', test_size=0.15, window_size=time_window, mode='running_window', shuffle=True)
    #train_df, test_df = shuffle_and_split(full_quantized_df, test_size=0.2)
    train.to_csv("IMU_Streams/preprocessed/train_samples_nobrick_filtered_peaks.csv")
    test.to_csv("IMU_Streams/preprocessed/test_samples_nobrick_filtered_peaks.csv")
    print('Done!')
