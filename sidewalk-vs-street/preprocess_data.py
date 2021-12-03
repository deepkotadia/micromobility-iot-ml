from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


COL_NAMES =['time',
           'accl_id', 'accl_x', 'accl_y', 'accl_z',
           'gyro_id', 'gyro_x', 'gyro_y', 'gyro_z',
           'mag_id', 'mag_x', 'mag_y', 'mag_z']


def read_imu_stream_file(filepath):
    df = pd.read_csv(filepath_or_buffer=filepath, names=COL_NAMES)
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
    normalized_df = (df - mean) / std
    return normalized_df


def read_all_stream_files_in_dir(dir_path, window_size=150):
    """
    reads all data streams as csv, normalize, divide into training samples and label
    prints data stats and returns dataframe of all training samples
    """
    filenames = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]  # get all stream filenames

    col_names = list()
    for i in range(window_size):
        col_names.extend([f'accl_x_{i}', f'accl_y_{i}', f'accl_z_{i}'])

    # Different types of streets and sidewalks (for secondary labels)
    df_sidewalk1 = pd.DataFrame()
    df_sidewalk = pd.DataFrame()
    df_street1 = pd.DataFrame()
    df_street2 = pd.DataFrame()
    df_street3 = pd.DataFrame()

    for filename in filenames:
        data_stream, num_rows = read_imu_stream_file(f'{dir_path}/{filename}')

        # append to corresponding df
        if 'sidewalk1' in filename:
            if df_sidewalk1.empty:
                df_sidewalk1 = data_stream
            else: 
                df_sidewalk1 = df_sidewalk1.append(data_stream, ignore_index=True)
        elif 'sidewalk' in filename:
            if df_sidewalk.empty:
                df_sidewalk = data_stream
            else:
                df_sidewalk = df_sidewalk.append(data_stream, ignore_index=True)
        elif 'street1' in filename or 'st1' in filename:
            if df_street1.empty:
                df_street1 = data_stream
            else:
                df_street1 = df_street1.append(data_stream, ignore_index=True)
        elif 'street2' in filename or 'st2' in filename:
            if df_street2.empty:
                df_street2 = data_stream
            else:
                df_street2 = df_street2.append(data_stream, ignore_index=True)
        elif 'street3' in filename or 'st3' in filename:
            if df_street3.empty:
                df_street3 = data_stream
            else:
                df_street3 = df_street3.append(data_stream, ignore_index=True)

    # normalize all dfs
    full_df = pd.concat((df_sidewalk1, df_sidewalk, df_street1, df_street2, df_street3), axis=0)
    mean = full_df.mean(skipna=True)
    std = full_df.std(skipna=True)
    normalized_sidewalk1 = normalize(df_sidewalk1, mean, std)
    normalized_sidewalk = normalize(df_sidewalk, mean, std)
    normalized_street1 = normalize(df_street1, mean, std)
    normalized_street2 = normalize(df_street2, mean, std)
    normalized_street3 = normalize(df_street3, mean, std)

    # extract features
    sidewalk1_samples = samples_and_feature_extraction(normalized_sidewalk1)
    sidewalk_samples = samples_and_feature_extraction(normalized_sidewalk)
    street1_samples = samples_and_feature_extraction(normalized_street1)
    street2_samples = samples_and_feature_extraction(normalized_street2)
    street3_samples = samples_and_feature_extraction(normalized_street3)

    # add secondary labels
    sidewalk1_samples['sublabel'] = 'sidewalk1'
    sidewalk_samples['sublabel'] = 'sidewalk'
    street1_samples['sublabel'] = 'street1'
    street2_samples['sublabel'] = 'street2'
    street3_samples['sublabel'] = 'street3'

    # combine different sidewalk and street dfs respectively
    all_sidewalk_samples = pd.concat((sidewalk1_samples, sidewalk_samples), axis=0)
    all_street_samples = pd.concat((street1_samples, street2_samples, street3_samples), axis=0)

    all_sidewalk_samples['label'] = 0
    all_street_samples['label'] = 1
    all_samples = pd.concat((all_sidewalk_samples, all_street_samples), axis=0)

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
