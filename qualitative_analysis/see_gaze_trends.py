import pandas as pd
import os
import matplotlib.pyplot as plt
import  numpy as np

data_folder = "/home/dp7972/Desktop/ASD_TD_WORk/ASD_SELF/data/Final_Ds12"
print(os.listdir(data_folder))

train_path = os.path.join(data_folder, "train")
test_path = os.path.join(data_folder, "test")
count, count2 = 0, 0
asd_count, td_count = 0, 0
asd_left_val,asd_right_val, td_left_val, td_right_val = [],[],[],[]
for tr_usr in os.listdir(train_path):
    usr_pth = os.path.join(train_path, tr_usr)
    td = 'TD' in tr_usr
    color = 'red'
    if td: color = 'green'
    for dta in os.listdir(usr_pth):
        dta_pth = os.path.join(usr_pth, dta)
        data = pd.read_csv(dta_pth)
        right_gaze_data = data['RPD'].to_numpy()
        left_gaze_data = data['LPD '].to_numpy()
        time = data['Time']
        gaze_data = right_gaze_data
        if 'ASD' in tr_usr:
            asd_left_val.append(np.mean(left_gaze_data))
            asd_right_val.append(np.mean(right_gaze_data))
            asd_count += 1
        if 'TD' in tr_usr:
            td_left_val.append(np.mean(left_gaze_data))
            td_right_val.append(np.mean(right_gaze_data))
            td_count += 1

        if 'ASD' in tr_usr and count == 0:
            count += 1
            # plt.plot(time, gaze_data, color = color, label = tr_usr)
            plt.hlines(xmin=min(time),xmax=max(time),y = np.mean( gaze_data), color=color, label=tr_usr)
        if 'TD' in tr_usr and count2 == 0:
            count2 += 1
            # plt.plot(time, gaze_data, color = color, label = tr_usr)
            plt.hlines(xmin=min(time),xmax=max(time),y = np.mean( gaze_data), color=color, label=tr_usr)

        else:
            plt.plot(time, gaze_data, color = color)
            plt.hlines(xmin=min(time),xmax=max(time),y = np.mean( gaze_data), color=color)



asd_left = np.var(asd_left_val)
asd_right = np.var(asd_right_val)

td_left = np.var(td_left_val)
td_right = np.var(td_right_val)

print("ASD Variance (L, R) : ", asd_left, asd_right)
print("TD Variance (L, R) : ", td_left, td_right)


#Trivial Classifier
thr = 5.00
count, count2 = 0, 0
asd_count, td_count = 0, 0
asd_left_val,asd_right_val, td_left_val, td_right_val = 0, 0, 0, 0
for tr_usr in os.listdir(test_path):
    usr_pth = os.path.join(test_path, tr_usr)
    td = 'TD' in tr_usr
    color = 'red'
    if td: color = 'green'
    for dta in os.listdir(usr_pth):
        dta_pth = os.path.join(usr_pth, dta)
        data = pd.read_csv(dta_pth)
        right_gaze_data = data['RPD'].to_numpy()
        left_gaze_data = data['LPD '].to_numpy()
        time = data['Time']
        gaze_data = right_gaze_data
        if 'ASD' in tr_usr:
            asd_left_val += 1*(np.mean(left_gaze_data)>thr)
            asd_right_val += 1*(np.mean(right_gaze_data)>thr)
            asd_count += 1
        if 'TD' in tr_usr:
            td_left_val += 1*(np.mean(left_gaze_data)<=thr)
            td_right_val += 1*(np.mean(right_gaze_data)<=thr)
            td_count += 1


asd_left = asd_left_val/asd_count
asd_right = asd_right_val/asd_count

td_left = td_left_val/td_count
td_right = td_right_val/td_count

print("ASD Test Accuracy (L, R) : ", asd_left, asd_right)
print("TD Test Accuracy (L, R) : ", td_left, td_right)

# print(data['RPD'].to_numpy())
plt.legend()
plt.title("Training users Right PD")

plt.show()

count, count2 = 0, 0
asd_count, td_count = 0, 0
for tr_usr in os.listdir(test_path):
    usr_pth = os.path.join(test_path, tr_usr)
    td = 'TD' in tr_usr
    color = 'red'
    if td: color = 'green'
    for dta in os.listdir(usr_pth):
        dta_pth = os.path.join(usr_pth, dta)
        data = pd.read_csv(dta_pth)
        # print(data.columns)
        right_gaze_data = data['RPD'].to_numpy()
        left_gaze_data = data['LPD '].to_numpy()
        time = data['Time']
        gaze_data = right_gaze_data

        if 'ASD' in tr_usr and count == 0:
            count += 1
            # plt.plot(time, gaze_data, color = color, label = tr_usr)
            plt.hlines(xmin=min(time),xmax=max(time),y = np.mean( gaze_data), color=color, label=tr_usr)

        if 'TD' in tr_usr and count2 == 0:
            count2 += 1
            # plt.plot(time, gaze_data, color = color, label = tr_usr)
            plt.hlines(xmin=min(time),xmax=max(time),y = np.mean( gaze_data), color=color, label=tr_usr)
        else:
            # plt.plot(time, gaze_data, color = color)
            plt.hlines(xmin=min(time),xmax=max(time),y = np.mean( gaze_data), color=color)


plt.legend()
plt.title("Test users Right PD")

plt.show()