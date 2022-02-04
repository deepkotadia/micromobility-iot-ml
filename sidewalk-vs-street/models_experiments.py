from preprocess_data import *
from models import *

data = read_all_stream_files_in_dir("IMU_Streams", window_size=75)

train, test = shuffle_and_split(data, test_size=0.2)

X_train = train.iloc[:,:-2]

y_train = train.iloc[:,-1]

X_test = test.iloc[:,:-2]

y_test = test.iloc[:,-1]

run_all_model_cross_val_stats(X_train, y_train)