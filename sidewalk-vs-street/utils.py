from unittest import skip
import numpy as np
import scipy
from preprocess_data import read_imu_stream_file
import os
import pandas as pd

COL_NAMES =['time',
           'accl_id', 'accl_x', 'accl_y', 'accl_z',
           'gyro_id', 'gyro_x', 'gyro_y', 'gyro_z',
           'mag_id', 'mag_x', 'mag_y', 'mag_z'] 

def compute_normalize_constants(dir):
    files = [os.path.join(dir,filename) for filename in os.listdir(dir) if os.path.isfile(os.path.join(dir,filename))]
    all_data = pd.DataFrame(columns = COL_NAMES)
    for file in files:
        data, _ = read_imu_stream_file(file)
        all_data = all_data.append(data)
    mean = np.nanmean(all_data[['accl_x', 'accl_y', 'accl_z', 'gyro_x', 'gyro_y', 'gyro_z']], axis=0).reshape(1,-1)
    std = np.nanstd(all_data[['accl_x', 'accl_y', 'accl_z', 'gyro_x', 'gyro_y', 'gyro_z']], axis=0).reshape(1,-1)
    print(mean.shape)
    print(std.shape)
    np.savetxt('constants.csv', np.concatenate((mean, std), axis=0), delimiter=',')



if __name__ == '__main__':
    dir = 'IMU_Streams'
    compute_normalize_constants(dir)
