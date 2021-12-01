from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
from scipy.sparse import data
from sklearn.model_selection import train_test_split


COL_NAMES =['time',
           'accl_id', 'accl_x', 'accl_y', 'accl_z',
           'gyro_id', 'gyro_x', 'gyro_y', 'gyro_z',
           'mag_id', 'mag_x', 'mag_y', 'mag_z']


def read_imu_stream_file(filepath):
    df = pd.read_csv(filepath_or_buffer=filepath, names=COL_NAMES)
    full_data_stream = df[['accl_x', 'accl_y', 'accl_z', 'gyro_x', 'gyro_y', 'gyro_z']]
    #gyro_df = df[['gyro_x', 'gyro_y', 'gyro_z']]
    #mag_df = df[['mag_x', 'mag_y', 'mag_z']]
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

def normalize(full_df_street, full_df_sidewalk, mean, std):
    '''z normalization (mean 0, std 1)'''
    mean1 = full_df_sidewalk.mean()
    mean2 = full_df_street.mean()
    std1 =  full_df_sidewalk.std()
    std2 = full_df_street.std()
    norm_full_df_street = (full_df_street-mean)/std
    norm_full_df_sidewalk = (full_df_sidewalk-mean)/std
    '''sanity check'''
    mean_post = norm_full_df_sidewalk.mean()
    street_mean_post = norm_full_df_street.mean()
    sidewalk_std_post = norm_full_df_sidewalk.std()
    street_std_post = norm_full_df_street.std()

    return norm_full_df_street, norm_full_df_sidewalk

    

def read_all_stream_files_in_dir(dir_path, window_size=150):
    '''reads all data streams as csv, normalize, divide into training samples and label
        prints data stats and returns dataframe of all training samples'''
    stats = dict()
    stats['total_rows_all_stream_files'] = 0
    filenames = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]  # get all stream filenames
    stats['total_stream_files'] = len(filenames)
    stats['sidewalk_files'], stats['street_files'] = 0, 0

    full_quantized_df = pd.DataFrame()  # dataframe with quantized data from all IMU streams
    col_names = list()
    for i in range(window_size):
        col_names.extend([f'accl_x_{i}', f'accl_y_{i}', f'accl_z_{i}'])
    full_df_street= pd.DataFrame()
    full_df_sidewalk = pd.DataFrame()
    for filename in filenames:
        data_stream, num_rows = read_imu_stream_file(f'{dir_path}/{filename}')
        stats['total_rows_all_stream_files'] += num_rows
        if 'sidewalk' in filename:
            label = 'sidewalk'
            stats['sidewalk_files'] += 1
            if full_df_sidewalk.empty:
                full_df_sidewalk = data_stream
            else: 
                full_df_sidewalk = full_df_sidewalk.append(data_stream)
        else:
            label = 'street'
            stats['street_files'] += 1
            if full_df_street.empty: 
                full_df_street = data_stream
            else:
                full_df_street = full_df_street.append(data_stream)
    #stats['sidewalk_files'] = len(full_df_sidewalk)
    #stats['street_files'] = len(full_df_street)
    full_df = pd.concat((full_df_sidewalk, full_df_street), axis=0)
    shape = full_df.shape
    mean = full_df.mean(skipna=True)
    std = full_df.std(skipna=True)
    stats['mean_values'] = mean
    stats['std_values'] = std
    normalized_street, normalized_sidewalk = normalize(full_df_street, full_df_sidewalk, mean, std)
    #full_df_street['label'] = 1
    #full_df_sidewalk['label'] = 0
        # create quantized df using only accelerometer data for now
    '''accl_quantized_df = quantize_and_clean(accl_df.to_numpy(), window_size, col_names, label)
        if full_quantized_df.empty:
            full_quantized_df = accl_quantized_df
        else:
            full_quantized_df = full_quantized_df.append(accl_quantized_df, ignore_index=True)

    stats['quantized_df_total_rows'] = full_quantized_df.shape[0]
    stats['quantized_df_total_cols'] = full_quantized_df.shape[1]'''

    print(stats)

    street_samples = samples_and_feature_extraction(normalized_street)
    sidewalk_samples = samples_and_feature_extraction(normalized_sidewalk)
    street_samples['label'] = 1
    sidewalk_samples['label'] = 0
    all_samples = pd.concat((street_samples, sidewalk_samples), axis=0)
    dims = all_samples.shape
    return all_samples
    


def shuffle_and_split(df, test_size=0.2):
    train, test = train_test_split(df, test_size=test_size, shuffle=True)
    return train, test


def samples_and_feature_extraction(dataframe, window_size = 150):
    '''divides into training points of size window_size (default 150 samples = 3 seconds)
        computes features (mean, std, percentiles)
        returns dataframe of dim (num_samples,num_features (24))'''
    col_names = ['mean_accl_x', 'mean_accl_y', 'mean_accl_z', 'mean_gyro_x', 'mean_gyro_y', 
    'mean_gyro_z', 'std_accl_x', 'std_accl_y', 'std_accl_z', 'std_gyro_x', 'std_gyro_y', 'std_gyro_z', 
    '90th_accl_x', '90th_accl_y', '90th_accl_z', '90th_gyro_x', '90th_gyro_y', '90th_gyro_z', 
    '10th_accl_x', '10th_accl_y', '10th_accl_z', '10th_gyro_x', '10th_gyro_y', '10th_gyro_z']
    idx = dataframe.shape[0] - (dataframe.shape[0] % window_size)
    dataframe = dataframe[:idx]
    num_samples = dataframe.shape[0]/window_size
    splits = np.array(np.vsplit(dataframe, num_samples))
    mean = np.nanmean(splits, axis=1)
    std = np.nanstd(splits, axis=1)
    percentile_90th = np.nanpercentile(splits, q=90, axis=1)
    percentile_10th = np.nanpercentile(splits, q=10, axis=1)
    samples = np.hstack((mean, std, percentile_90th, percentile_10th))
    all_samples = pd.DataFrame(samples, columns=col_names)
    return all_samples




if __name__ == '__main__':
    full_quantized_df = read_all_stream_files_in_dir('IMU_Streams')
    train_df, test_df = shuffle_and_split(full_quantized_df, test_size=0.2)
    print('Done!')
