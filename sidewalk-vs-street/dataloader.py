from random import sample

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from os.path import join
from os import listdir
import numpy as np
from preprocess_data import read_imu_stream_file

class SidewalkDataSet(torch.utils.data.Dataset):

    def __init__(self, path, window_size, step_size=1):
        super().__init__()
        self.window_size = window_size
        self.data = [join(path, f) for f in listdir(path)]
        
        self.step_size = step_size
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path = self.data[i]
        if 'sidewalk' in path:
            label = 1
        else:
            label = 0
        dataframe, _ = read_imu_stream_file(path)
        idx = dataframe.shape[0] - (dataframe.shape[0] % self.step_size)
        dataframe = dataframe[:idx]
        dataframe = dataframe[['accl_x', 'accl_y', 'accl_z']]
        sample = []
        for i in range(self.window_size, len(dataframe), self.step_size):
            sample.append(dataframe.iloc[i-self.window_size:i].to_numpy())
        labels = torch.tensor([label for i in range(len(sample))]).reshape(-1,1)
        labels = F.one_hot(labels, num_classes=2)
        sample = np.rollaxis(np.array(sample), 2, 1)
        sample = torch.tensor(np.array(sample))
        return sample, labels

    
if __name__ == "__main__":
    path = "IMU_Streams/train"
    dataset = SidewalkDataSet(path, window_size=256)
    print(len(dataset))

    sample = dataset[0]


        



