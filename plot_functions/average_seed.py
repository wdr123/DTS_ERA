import os

import numpy as np
import pandas as pd


def average_seed(dir_path):
    save_lstm_all = []

    files = os.listdir(dir_path)
    for file in files:
        if file[:4] == 'Save':
            file_path = os.path.join(dir_path, file)
            data = pd.read_csv(file_path, header=0)
            # print(type(data.values))
            # print(type(data.columns))
            data_numpy = np.array(data.values,dtype=float)
            save_lstm_all.append(data_numpy)

    average_data = np.mean(save_lstm_all, axis=0)
    # print(type(average_data))
    # print(average_data)
    average_df = pd.DataFrame(average_data, columns=data.columns, index=None)
    if not os.path.exists(os.path.join(dir_path,'average_seed_RAM.csv')):
        average_df.to_csv(os.path.join(dir_path,'average_seed_RAM.csv'))
    return average_df
