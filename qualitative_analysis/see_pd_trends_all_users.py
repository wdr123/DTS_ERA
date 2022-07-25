import pandas as pd
import os
import matplotlib.pyplot as plt
import  numpy as np

data_folder = "/home/dp7972/Desktop/ASD_TD_WORk/ASD_SELF/data/Final_Ds12"
print(os.listdir(data_folder))

train_path = os.path.join(data_folder, "train")
test_path = os.path.join(data_folder, "test")


def plot_user_lines(path):
    count, count2 = 0, 0
    users = os.listdir(path)
    for usr in users:
        usr_pth = os.path.join(path, usr)
        td = 'TD' in usr
        color = 'red'
        if td: color = 'green'
        for dta in os.listdir(usr_pth):
            dta_pth = os.path.join(usr_pth, dta)
            data = pd.read_csv(dta_pth)
            right_gaze_data = data['RPD'].to_numpy()
            left_gaze_data = data['LPD '].to_numpy()
            time = data['Time']
            gaze_data = left_gaze_data

            if 'ASD' in usr and count == 0:
                count += 1
                # plt.plot(time, gaze_data, color = color, label = tr_usr)
                plt.hlines(xmin=min(time),xmax=max(time),y = np.mean( gaze_data), color=color, label=usr)
            if 'TD' in usr and count2 == 0:
                count2 += 1
                # plt.plot(time, gaze_data, color = color, label = tr_usr)
                plt.hlines(xmin=min(time),xmax=max(time),y = np.mean( gaze_data), color=color, label=usr)

            else:
                # plt.plot(time, gaze_data, color = color)
                plt.hlines(xmin=min(time),xmax=max(time),y = np.mean( gaze_data), color=color)


# print(data['RPD'].to_numpy())
plot_user_lines(train_path)
plot_user_lines(test_path)
plt.legend()
plt.title("All users Left PD")

plt.show()
