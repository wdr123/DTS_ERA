import pandas as pd
import numpy as np
import os


df1 = pd.read_csv("../results/visualization/origin/ASD('02',)_gamelevel_4_origin.csv", header=None)
df2 = pd.read_csv("../results/visualization/origin/TD('05',)_gamelevel_4_origin.csv", header=None)

df1.insert(0, 'T', 268)
df1.insert(0, 'G', 4)
df1.insert(0, 'U', 2)


df2.insert(0, 'T', 180)
df2.insert(0, 'G', 4)
df2.insert(0, 'U', 5)

data_destination_folder = '../results/visualization/traj'

if not os.path.exists(data_destination_folder):
    os.mkdir(data_destination_folder)

data_destination1 = os.path.join(data_destination_folder, "ASD('02',)_gamelevel_4_traj.csv")
data_destination2 = os.path.join(data_destination_folder, "TD('05',)_gamelevel_4_traj.csv")

df1.to_csv(data_destination1, index=None, header=None)
df2.to_csv(data_destination2, index=None, header=None)