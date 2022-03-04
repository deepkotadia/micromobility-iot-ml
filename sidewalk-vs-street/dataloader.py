from random import sample
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from os.path import join
from os import listdir
import numpy as np
from preprocess_data import read_imu_stream_file

class SidewalkDataSet(torch.utils.data.Dataset):

    def __init__(self, path, path_to_normalize_constants, columns=['accl_x', 'accl_y', 'accl_z']):
        super().__init__()
        #.window_size = window_size
        self.data = [join(path, f) for f in listdir(path)]
        self.normalize_constants = pd.read_csv(path_to_normalize_constants, index_col=0)
        self.columns = columns
        self.median = self.normalize_constants.loc[columns, ['median']].to_numpy().reshape(1,-1)
        self.intQr = self.normalize_constants.loc[columns, ['intQrange']].to_numpy().reshape(1,-1)
        #self.step_size = step_size
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path = self.data[i]

        dataframe = pd.read_csv(path)
        label = F.one_hot(torch.tensor(dataframe['label'])[0], num_classes=2).float()
        sample = torch.tensor(self.normalize(dataframe[self.columns]))
        sample = torch.permute(sample, dims=(1, 0))
        return sample, label

    def normalize(self,sample):
        sample = sample.to_numpy()
        sample = (sample-self.median)/self.intQr
        return sample
    
if __name__ == "__main__":
    path = "IMU_Data/train"
    path_to_constants ='IMU_Data/data_stats_train.csv'
    dataset = SidewalkDataSet(path, path_to_constants)
    print(len(dataset))

    sample = dataset[0]
    print(sample[1])


        



