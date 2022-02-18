from cmath import cos, sin
from matplotlib import markers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy.signal import savgol_filter, find_peaks, spectrogram, lfilter, butter, sosfilt
from scipy.fftpack import fft, fftfreq
from os.path import *
from os import listdir, mkdir
from preprocess_data import read_imu_stream_file
import pywt

cols = ['mean_accl_x', 'mean_accl_y', 'mean_accl_z',
       'mean_gyro_x', 'mean_gyro_y', 'mean_gyro_z', 'std_accl_x', 'std_accl_y',
       'std_accl_z', 'std_gyro_x', 'std_gyro_y', 'std_gyro_z', '90th_accl_x',
       '90th_accl_y', '90th_accl_z', '90th_gyro_x', '90th_gyro_y',
       '90th_gyro_z', '10th_accl_x', '10th_accl_y', '10th_accl_z',
       '10th_gyro_x', '10th_gyro_y', '10th_gyro_z', 'range_accl_x',
       'range_accl_y', 'range_accl_z', 'range_gyro_x', 'range_gyro_y',
       'range_gyro_z', 'sublabel', 'label']

ALL_COL_NAMES =['time',
           'accl_id', 'accl_x', 'accl_y', 'accl_z',
           'gyro_id', 'gyro_x', 'gyro_y', 'gyro_z',
           'mag_id', 'mag_x', 'mag_y', 'mag_z']

FEATURE_COLS = ['accl_x', 'accl_y', 'accl_z', 'gyro_x', 'gyro_y', 'gyro_z']

def viz_peaks(signal):
    data = signal['accl_y']
    
    filtered = savgol_filter(data, polyorder=3, window_length=75 )[:150]
    #height_min = np.percentile(filtered, q=0.9)
    #height_max = np.percentile(filtered, q=0.75)
    peak_idx, properties = find_peaks(filtered, prominence=0.03)
    my_peaks = filtered[peak_idx]
    dummy = np.zeros((150))
    y = np.arange(150)
    dummy[peak_idx] = filtered[peak_idx]
    print(properties)
    plt.plot(filtered, color='blue')
    plt.scatter(x=y, y=dummy, marker='x', color='red')
    plt.ylim([np.min(filtered)-0.1, np.max(filtered)+0.1])
    plt.show()

def filter_all_axes(signal):
    filtered = savgol_filter(signal, window_length=75, polyorder=3, axis=0)
    filtered_df = pd.DataFrame(filtered, columns=FEATURE_COLS)
    return filtered_df

def lowpass_filter(signal):
    filtered_df = pd.DataFrame(lfilter(b=[1], a= [0.96], x=signal, axis=0), columns=FEATURE_COLS)
    return filtered_df

def define_filter():
    return butter(1, Wn=0.02, btype='high', output='sos')

def high_pass_filter(signal, sos_filter):
    filtered = sosfilt(sos_filter, signal)
    return filtered



def fft_viz(signal, figtitle, savedir, timewindow=10):
    frequency = 50
    num_samples = 50*timewindow
    windowlength = num_samples
    time_step = 1/frequency
    fig, ax = plt.subplots(6,2, figsize=(20,20))
    ax1, ax2 = ax[:,0], ax[:,1]
    
    for idx, feat in enumerate(FEATURE_COLS):
        axis = signal.loc[:,[feat]][num_samples:num_samples+num_samples] #from middle-ish
        fft_signal = fft(axis)
        freq_bins = fftfreq(num_samples, d=time_step)[:int(num_samples//2)]
        y = 2.0/frequency * np.abs(fft_signal)[:int(num_samples/2)]
        ax1[idx].plot(axis)
        try:
            ax2[idx].plot(freq_bins, y)
        except ValueError:
            print(figtitle, f'length of signal: {axis.shape}, length of fft {fft_signal.shape}')
        ax1[idx].set_title(label='raw signal '+feat)
        ax2[idx].set_title('fft signal {}'.format(feat))
        ax2[idx].set_ylabel('strength')
        ax2[idx].set_xlabel('spectra')
        
    #plt.show()
    plt.savefig(join(savedir, figtitle+'.png'))
    plt.close()

def spectrogram_viz(signal, figtitle, save_dir):
    fig, ax = plt.subplots(3, 2, figsize=(20,20))
    ax = ax.flatten(order='F')
    for idx, feat in enumerate(FEATURE_COLS):
        axis = signal[feat]
        f, ts, spectr = spectrogram(axis, fs=50)
        spectr =  spectr/np.max(spectr)
        ax[idx].pcolormesh(ts, f, spectr)
        ax[idx].set_title(label=feat)
        ax[idx].set_ylabel('spectrum')
        ax[idx].set_xlabel('time')

    #plt.show()
    #plt.savefig(f'{save_dir}/{figtitle}.png')
    for axs in ax:
        axs.set_ylim(0, 15)
    plt.savefig(f'{save_dir}/{figtitle}_axlim.png')
    plt.close()


def dummytest():
    N = 600
# sample spacing
    T = 1.0 / 800.0
    x = np.linspace(0.0, N*T, N, endpoint=False)
    y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
    yf = fft(y)
    xf = fftfreq(N, T)[:N//2]
    plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
    plt.show()

def dummy_step_func():
    step_func = np.array([-1 for i in range(200)] + [1 for i in range(200)])
    step = 5
    speeds = [0]
    dt = 1/50
    for i in range(step, len(step_func), step):

        speeds.append(speeds[-1] + np.sum(step_func[i-step:i]*dt))
    plt.subplot(1,2,1)
    plt.plot(step_func)
    plt.subplot(1,2,2)
    plt.plot(speeds)
    plt.show()
'''float rotationMatrix[3][3] = 
{
  { cos(alpha)*cos(beta) , cos(alpha)*sin(beta)*sin(theta) - sin(alpha)*cos(theta) , cos(alpha)*sin(beta)*cos(theta) + sin(alpha)*sin(theta)},
  { sin(alpha)*cos(beta) , sin(alpha)*sin(beta)*sin(theta) + cos(alpha)*cos(theta) , sin(alpha)*sin(beta)*cos(theta) - cos(alpha)*sin(theta)},
  {     -1* sin(beta)    ,                  cos(beta) * sin(theta)                 ,               cos(beta) * cos(theta)                   }
};'''
def getRotationMatrix(gyro_est):
    theta, beta, alpha = gyro_est[0], gyro_est[1], gyro_est[2]
    rotationMatrix = np.array(
        [[cos(alpha)*cos(beta), cos(alpha)*sin(beta)*sin(theta)-sin(alpha)*cos(theta), cos(alpha)*sin(beta)*cos(theta)+sin(alpha)*sin(theta)],
        [sin(alpha)*cos(beta), sin(alpha)*sin(beta)*sin(theta)+cos(alpha)*cos(theta), sin(alpha)*sin(beta)*cos(theta) - cos(alpha)*sin(theta)],
        [-1*sin(beta), cos(beta)*sin(theta), cos(beta)*cos(theta)]])
    return rotationMatrix

def rotate_acceleration(accl, rotationMatrix):
    rotated = np.dot(accl, rotationMatrix)
    print(rotated)
    return rotated

def estimate_gyro(gyro):
    return NotImplemented

def correct_for_gravity(accl_rotated):
    gravity = np.array([0, 0, 1])
    return accl_rotated - gravity

def compute_speed(forward_accl):
    dt =  1/50 #sampling rate
    step = 25 #25 samples is 0.5 seconds
    error_est = np.sum(forward_accl[:step]*dt)
    speeds = [0]
    for i in range(step, len(forward_accl), step):
        speeds.append(speeds[-1] + np.sum(forward_accl[i-step:i]*dt - error_est))
    return speeds
    plt.plot(speeds)
    plt.savefig(name+'speed.png')
    #plt.show()
    plt.close()
    
def compensate_gravity_compute_speed_plot(signal, filename):
    fig, ax = plt.subplots(5, 1, figsize=(20, 30))
    sos_filt = define_filter()
    filt_x =  high_pass_filter(signal['accl_x'], sos_filt)
    filt_y = high_pass_filter(signal['accl_y'], sos_filt)
    filt_z = high_pass_filter(signal['accl_z'], sos_filt)
    speeds = speed(filt_z)
    reg_speed = speed(signal['accl_z'])
    ax[0].plot(filt_x)
    ax[0].set_title('filtered x')
    ax[1].plot(filt_y)
    ax[1].set_title('filtered y')
    ax[2].plot(filt_z)
    ax[2].set_title('filtered z')
    ax[3].plot(speeds)
    ax[3].set_title(' filtered estimated speed')
    ax[4].plot(reg_speed)
    ax[4].set_title('non filtered speed ')
    #plt.show()
    plt.savefig(name+'speed_grav.png')
    plt.close()




def speed(signal):

    dt = 1/50

    speeds = [0]
    error_est = np.sum(signal[:25]*dt)
    step = int((0.5*50))
    for i in range(step, signal.shape[0], step):
        #v0 = np.sum(signal[(i-step-int(step/4)):(i-step)]*dt-error_est)
        speeds.append(speeds[-1] + np.sum(signal[i-step:i]*dt)-error_est)
    return speeds


    #segments = np.array(np.split(signal, segments))
    #speed = np.sum(segments*dt, axis=1)
    #speed = speed - reset
    plt.plot(speeds)
    plt.savefig(name+'speed.png')
    #plt.show()
    plt.close()





if __name__ == "__main__":
    #dummytest()
    #dummy_step_func()

    
    savedir = 'plots_fft/filtered/lowpass'
    savedir_spectr = 'plots_fft/spectrogram/lowpass'
    if not isdir(savedir_spectr):
        mkdir(savedir_spectr)
    if not isdir(savedir):
        mkdir(savedir)
    files = [f'IMU_Streams/{f}' for f in listdir('IMU_Streams') if isfile(join('IMU_Streams', f))]
    print('plotting all signals')
    for file in files:
        
        signal, _ = read_imu_stream_file(file)
        
        name = splitext(split(file)[1])[0]
        compensate_gravity_compute_speed_plot(signal, name)
        '''
        speed(signal['accl_z'].to_numpy(), window_size=3)
        
        
        signal.interpolate(method='linear', limit=2, limit_direction='both', axis=0, inplace=True)
        
        
        #signal = filter_all_axes(signal)
        signal = lowpass_filter(signal)
        spectrogram_viz(signal, name, savedir_spectr)
        fft_viz(signal, figtitle=name, savedir=savedir, timewindow=15)
        '''
        
    print('done!')
    #signal = pd.read_csv('IMU_Streams/st1MedDNoweight1.csv', names=ALL_COL_NAMES)

    #viz_peaks(signal)
    