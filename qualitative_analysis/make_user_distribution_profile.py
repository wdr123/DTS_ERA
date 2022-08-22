import os
import numpy as np
import pandas as pd

path = os.getcwd()

one_level_up = '/'.join(path.split('/')[:-1])

data_path = os.path.join(one_level_up, "data")

user_np = np.zeros((20,5))
for dp in os.listdir(data_path):
    if 'Final_Ds' not in dp:
        continue
    test_path = os.path.join(os.path.join(data_path, dp), "test")
    test_users = os.listdir(test_path)

    td = [int(u[-2:]) for u in test_users if 'TD' in u]
    asd = [int(u[-2:]) for u in test_users if 'ASD' in u]

    seed = int(dp.split("Final_Ds")[-1])
    user_np[seed-1] = np.array([seed, asd[0], asd[1],td[0],td[1]], dtype=int)

# print(user_np)

column_names = ['Seed', 'ASD1', 'ASD2', 'TD1', 'TD2']

df = pd.DataFrame(data = user_np, columns=column_names, dtype=int)
df.to_csv("user_distribution_profile.csv")
print(df)