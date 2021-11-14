import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal


def plot_from_accelerometer_df(df, frequency, dataset_id, apply_filter=False):
    """
    Helper function to plot accelerometer graph given dataframe with readings
    """
    # Time values for x-axis
    df = df[:500]
    N = df.shape[0]  # number of samples
    x_axis = np.linspace(0, N/frequency, num=N)
    df.insert(0, 'norm_time', x_axis)

    if apply_filter:
        # Savgol Filter
        yhat = scipy.signal.savgol_filter(df['accl_y'].to_numpy(), window_length=21, polyorder=3)
        df.insert(2, 'filtered_y', yhat)

        # Find peaks in savgol filtered signal
        peaks, _ = scipy.signal.find_peaks(yhat, distance=30)
        print(len(peaks))

        # Plot Chart
        df.plot(x='norm_time', y=['filtered_y'], figsize=(10, 5))
        plt.xlabel('Time (s)')
        plt.ylabel('Accelerometer Readings (m/s^2)')
        plt.title('Accelerometer Readings for three axes over Time for {0}'.format(dataset_id))
        plt.plot(peaks / frequency, yhat[peaks], 'x')
        plt.show()

    else:
        # Plot Chart
        df.plot(x='norm_time', y=['accl_x', 'accl_y', 'accl_z'], figsize=(10, 5))
        plt.xlabel('Time (s)')
        plt.ylabel('Accelerometer Readings (m/s^2)')
        plt.title('Accelerometer Readings for three axes over Time for {0}'.format(dataset_id))
        plt.show()


# Read accelerometer data from all three CSV files
col_names=['time',
           'accl_id', 'accl_x', 'accl_y', 'accl_z',
           'gyro_id', 'gyro_x', 'gyro_y', 'gyro_z',
           'mag_id', 'mag_x', 'mag_y', 'mag_z']
a = pd.read_csv(filepath_or_buffer='IMU_Streams/sidewalk1MedDNoweight.csv', names=col_names)
b = pd.read_csv(filepath_or_buffer='IMU_Streams/st1MedDNoWeight1.csv', names=col_names)
c = pd.read_csv(filepath_or_buffer='IMU_Streams/street1MedBweight1.csv', names=col_names)

plot_from_accelerometer_df(a, 50, 'sidewalk Med Weight', apply_filter=True)
plot_from_accelerometer_df(b, 50, 'street Med Weight', apply_filter=True)
# plot_from_accelerometer_df(c, 50, 'street Med Weight')
