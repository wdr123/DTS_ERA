import os

import numpy as np
import pandas as pd

save_cat_all = []

files = os.listdir('../csv_file/save_cat')
for file in files:
    file_path = os.path.join('../csv_file/save_cat', file)
    data = pd.read_csv(file_path, header=0)
    # print(type(data.values))
    # print(type(data.columns))
    data_numpy = np.array(data.values,dtype=float)
    save_cat_all.append(data_numpy)

average_data = np.mean(save_cat_all, axis=0)
# print(type(average_data))
# print(average_data)
average_df = pd.DataFrame(average_data, columns=data.columns, index=None)
average_df.to_csv('average_seed_cat.csv')
