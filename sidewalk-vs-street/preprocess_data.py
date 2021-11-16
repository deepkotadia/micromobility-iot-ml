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
    accl_df = df[['accl_x', 'accl_y', 'accl_z']]
    gyro_df = df[['gyro_x', 'gyro_y', 'gyro_z']]
    mag_df = df[['mag_x', 'mag_y', 'mag_z']]
    return accl_df, gyro_df, mag_df, accl_df.shape[0]


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


def read_all_stream_files_in_dir(dir_path, window_size=150):
    stats = dict()
    stats['total_rows_all_stream_files'] = 0
    filenames = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]  # get all stream filenames
    stats['total_stream_files'] = len(filenames)
    stats['sidewalk_files'], stats['street_files'] = 0, 0

    full_quantized_df = pd.DataFrame()  # dataframe with quantized data from all IMU streams
    col_names = list()
    for i in range(window_size):
        col_names.extend([f'accl_x_{i}', f'accl_y_{i}', f'accl_z_{i}'])

    for filename in filenames:
        accl_df, gyro_df, mag_df, num_rows = read_imu_stream_file(f'{dir_path}/{filename}')
        stats['total_rows_all_stream_files'] += num_rows
        if 'sidewalk' in filename:
            label = 'sidewalk'
            stats['sidewalk_files'] += 1
        else:
            label = 'street'
            stats['street_files'] += 1

        # create quantized df using only accelerometer data for now
        accl_quantized_df = quantize_and_clean(accl_df.to_numpy(), window_size, col_names, label)
        if full_quantized_df.empty:
            full_quantized_df = accl_quantized_df
        else:
            full_quantized_df = full_quantized_df.append(accl_quantized_df, ignore_index=True)

    stats['quantized_df_total_rows'] = full_quantized_df.shape[0]
    stats['quantized_df_total_cols'] = full_quantized_df.shape[1]

    print(stats)

    return full_quantized_df


def shuffle_and_split(df, test_size=0.2):
    train, test = train_test_split(df, test_size=test_size, shuffle=True)
    return train, test


if __name__ == '__main__':
    full_quantized_df = read_all_stream_files_in_dir('IMU_Streams')
    train_df, test_df = shuffle_and_split(full_quantized_df, test_size=0.2)
    print('Done!')
