import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from average_seed import average_seed

result_test_dir = '../results/all/partial'
ASD_mean = []
ASD_std = []
TD_mean = []
TD_std = []

for usr_folder in os.listdir(result_test_dir):
    usr_path = os.path.join(result_test_dir, usr_folder)
    for partial_folder in os.listdir(usr_path):
        partial_path = os.path.join(usr_path, partial_folder)
        sd_mean, sd_std = average_seed(partial_path)
        sd_mean = np.array(sd_mean.loc[:,'If Correct']).squeeze()
        sd_std = sd_std[:,-3].squeeze()
        # print(sd_mean)
        # print(sd_std)

        if usr_folder == "ASD":
            ASD_mean.append(sd_mean)
            ASD_std.append(sd_std)
        elif usr_folder == "TD":
            TD_mean.append(sd_mean)
            TD_std.append(sd_std)

ASD_mean = np.array(ASD_mean)
ASD_std = np.array(ASD_std)
TD_mean = np.array(TD_mean)
TD_std = np.array(TD_std)

print(ASD_mean,ASD_std,TD_mean,TD_std)

# fig, ax = plt.subplots()
# fig1, ax1 = plt.subplots()
#
# clrs = sns.color_palette("husl", 2)
# with sns.axes_style("darkgrid"):
#     partial_list = np.arange(2,11,2) / 10
#     ax.plot(partial_list, ASD_mean, label='ASD', c=clrs[0])
#     ax.fill_between(partial_list, ASD_mean - ASD_std, ASD_mean + ASD_std, alpha=0.3, facecolor=clrs[0])
#     ax.legend()
#     ax1.plot(partial_list, TD_mean, label='TD', c=clrs[1])
#     ax1.fill_between(partial_list, TD_mean - TD_std, TD_mean + TD_std, alpha=0.3, facecolor=clrs[1])
#     ax1.legend()
#
# fig.show()
# fig1.show()