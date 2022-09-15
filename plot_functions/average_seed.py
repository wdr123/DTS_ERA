import os

import numpy as np
import pandas as pd


def average_seed(dir_path):
    save_lstm_all = []

    files = os.listdir(dir_path)
    for file in files:
        if 'Save' in file or 'Partial' in file:
            file_path = os.path.join(dir_path, file)
            data = pd.read_csv(file_path, header=0,)
            # print(type(data.values))
            # print(type(data.columns))
            data_numpy = np.array(data.values,dtype=float)
            # print(data_numpy)
            # print(data_numpy.shape)
            save_lstm_all.append(data_numpy)

    average_data = np.mean(save_lstm_all, axis=0)
    average_std = np.std(save_lstm_all, axis=0)
    # print(average_data.shape)
    # print(average_std.shape)
    average_df = pd.DataFrame(average_data, columns=data.columns, index=None)
    if not os.path.exists(os.path.join(dir_path,'average_seed_RAM.csv')):
        average_df.to_csv(os.path.join(dir_path,'average_seed_RAM.csv'),index=None)
    return average_df, average_std
