import pandas as pd
import os
import matplotlib.pyplot as plt
import  numpy as np
import matplotlib.cm as cm
import matplotlib
path = os.getcwd()

one_level_up = '/'.join(path.split('/')[:-1])

data_folder = one_level_up + "/data/Final_Ds18"
print(os.listdir(data_folder))

train_path = os.path.join(data_folder, "train")
test_path = os.path.join(data_folder, "test")
count, count2 = 0, 0
asd_count, td_count = 0, 0
asd_left_val,asd_right_val, td_left_val, td_right_val = [],[],[],[]
the_path = test_path
for tr_usr in os.listdir(the_path):
    usr_pth = os.path.join(the_path, tr_usr)
    td = 'TD' in tr_usr
    color = 'red'
    if td: color = 'green'
    for dta in os.listdir(usr_pth):
        dta_pth = os.path.join(usr_pth, dta)
        data = pd.read_csv(dta_pth)
        right_gaze_data = data['RPD'].to_numpy()
        left_gaze_data = data['LPD '].to_numpy()
        time = data['Time'].to_numpy()
        # print(data.columns)
        gaze_data = left_gaze_data
        gaze_data_x = data['LGPX'].to_numpy()
        gaze_data_y = data['LGPY'].to_numpy()
        max_t = max(time)
        min_t = min(time)
        norm = matplotlib.colors.Normalize(vmin=min_t, vmax=max_t)
        cmap = cm.hot
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        # for t, x, y in zip(time, gaze_data_x, gaze_data_y):

        print(tr_usr, dta)
        if min(gaze_data_x)<-0.1:
            print("Negative Left Gaze x user: ", tr_usr, dta, min(gaze_data_x))
        if min(gaze_data_y)<-0.3:
            print("Negative Left Gaze y user: ", tr_usr, dta,min(gaze_data_y))
        if max(gaze_data_x)>1.0:
            print("MAX Left Gaze x user: ", tr_usr, dta, max(gaze_data_x))
        if max(gaze_data_y)>1.0:
            print("MAX Left Gaze y user: ", tr_usr, dta, max(gaze_data_y))

        # plt.scatter(gaze_data_x,gaze_data_y, c=time, cmap="plasma")
        # plt.xlabel("x")
        # plt.ylabel("y")
        # plt.colorbar(label="Time", orientation="horizontal")
        # title = f"Left gaze User:{tr_usr} Level: {dta.split('merged')[1].split('.csv')[0]}"
        # plt.title(title)
        # plt.gca().invert_yaxis()
        # plt.savefig(title)
        # plt.show()

