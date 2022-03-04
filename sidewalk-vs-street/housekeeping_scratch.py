import pandas as pd
import numpy as np
constants = pd.read_csv("IMU_Streams/normalization_constants.csv")
print(constants.info())

columns = constants.iloc[:,0]
mean = constants['0'].to_numpy().reshape(1,-1)
std = constants['1'].to_numpy().reshape(1,-1)

const = np.concatenate((mean, std), axis=0)
save_file = pd.DataFrame(const, columns=columns).reset_index(drop=True)
save_file.to_csv('IMU_Streams/normalizing_constants.csv')

constants_file = pd.read_csv("IMU_Streams/normalizing_constants.csv")
#constants = constants.pivot(index='index', columns = 'Unnamed:0')
breakpoint
#constants.to_csv("constants.csv")