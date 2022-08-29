import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from data.merged_instance_generator_visualization import ASDTDTaskGenerator
from torch.utils.data import DataLoader
from utilFiles.the_args import get_args
from utilFiles.set_deterministic import make_deterministic

args, _ = get_args()
attn = True
T = 5
obs_window = 10
num_attn = 2
train_dl = DataLoader(ASDTDTaskGenerator("train", data_path="Fine_grain", args = args),batch_size=1,shuffle=True)
make_deterministic(seed=0)




for batch_idx, (touch_gaze_data, label, level, user_id) in enumerate(train_dl):
    whole_left_gaze_x_list_ASD = []
    whole_left_gaze_y_list_ASD = []
    whole_right_gaze_x_list_ASD = []
    whole_right_gaze_y_list_ASD = []
    whole_touch_x_list_ASD = []
    whole_touch_y_list_ASD = []
    whole_touch_hard_list_ASD = []
    whole_duration_ASD = []

    whole_left_gaze_x_list_TD = []
    whole_left_gaze_y_list_TD = []
    whole_right_gaze_x_list_TD = []
    whole_right_gaze_y_list_TD = []
    whole_touch_x_list_TD = []
    whole_touch_y_list_TD = []
    whole_touch_hard_list_TD = []
    whole_duration_TD = []

    label = label[0]
    level = level[0]
    touch_gaze_data = touch_gaze_data[0].cpu().numpy()
    time_lapse = touch_gaze_data[:,0]
    time_lapse_before = np.concatenate([np.array([0]),touch_gaze_data[:-1,0]],axis=0)
    touch_gaze_data = touch_gaze_data[:,1:]
    touch_gaze_data[touch_gaze_data <= 0] = 0.01
    touch_gaze_data_before = np.minimum(touch_gaze_data[:,:-1], 1)
    touch_gaze_data[:,:-1] = touch_gaze_data_before
    touch_gaze_data[:,-1] = touch_gaze_data[:,-1] / np.linalg.norm(touch_gaze_data[:,-1])

    attn_start = np.random.randint(low=len(touch_gaze_data) - obs_window, high=None, size=num_attn*T)
    attn_end = attn_start + obs_window

    if label==0:
        whole_left_gaze_x_list_TD = touch_gaze_data[:, 0]
        whole_left_gaze_y_list_TD = touch_gaze_data[:, 1]
        whole_right_gaze_x_list_TD = touch_gaze_data[:, 2]
        whole_right_gaze_y_list_TD = touch_gaze_data[:, 3]
        whole_touch_x_list_TD = touch_gaze_data[:, 4]
        whole_touch_y_list_TD = touch_gaze_data[:, 5]
        whole_touch_hard_list_TD = touch_gaze_data[:, 6]
        whole_duration_TD = time_lapse - time_lapse_before

        for idx, attn in enumerate(list(zip(attn_start, attn_end))):

            left_gaze_x_list_TD = touch_gaze_data[attn[0]:attn[1], 0]
            left_gaze_y_list_TD = touch_gaze_data[attn[0]:attn[1], 1]
            right_gaze_x_list_TD = touch_gaze_data[attn[0]:attn[1], 2]
            right_gaze_y_list_TD = touch_gaze_data[attn[0]:attn[1], 3]
            touch_x_list_TD = touch_gaze_data[attn[0]:attn[1], 4]
            touch_y_list_TD = touch_gaze_data[attn[0]:attn[1], 5]
            touch_hard_list_TD = touch_gaze_data[attn[0]:attn[1], 6]
            duration_TD = time_lapse[attn[0]:attn[1]] - time_lapse_before[attn[0]:attn[1]]

            table_TD = np.array(
                [left_gaze_x_list_TD, left_gaze_y_list_TD, right_gaze_x_list_TD, right_gaze_y_list_TD, touch_x_list_TD,
                 touch_y_list_TD, touch_hard_list_TD, duration_TD])
            df_TD = pd.DataFrame(table_TD)
            if not os.path.exists(f'results/visualization/TD{user_id}_gamelevel_{level}_attn'):
                os.mkdir(f'results/visualization/TD{user_id}_gamelevel_{level}_attn')
            df_TD.to_csv(f'results/visualization/TD{user_id}_gamelevel_{level}_attn/attention_{idx}.csv', index=None,
                         header=None)

    else:
        whole_left_gaze_x_list_ASD=touch_gaze_data[:, 0]
        whole_left_gaze_y_list_ASD=touch_gaze_data[:, 1]
        whole_right_gaze_x_list_ASD=touch_gaze_data[:, 2]
        whole_right_gaze_y_list_ASD=touch_gaze_data[:, 3]
        whole_touch_x_list_ASD=touch_gaze_data[:, 4]
        whole_touch_y_list_ASD=touch_gaze_data[:, 5]
        whole_touch_hard_list_ASD=touch_gaze_data[:, 6]
        whole_duration_ASD = time_lapse-time_lapse_before

        for idx, attn in enumerate(list(zip(attn_start, attn_end))):

            left_gaze_x_list_ASD=touch_gaze_data[attn[0]:attn[1], 0]
            left_gaze_y_list_ASD=touch_gaze_data[attn[0]:attn[1], 1]
            right_gaze_x_list_ASD=touch_gaze_data[attn[0]:attn[1], 2]
            right_gaze_y_list_ASD=touch_gaze_data[attn[0]:attn[1], 3]
            touch_x_list_ASD=touch_gaze_data[attn[0]:attn[1], 4]
            touch_y_list_ASD=touch_gaze_data[attn[0]:attn[1], 5]
            touch_hard_list_ASD=touch_gaze_data[attn[0]:attn[1], 6]
            duration_ASD=time_lapse[attn[0]:attn[1]] - time_lapse_before[attn[0]:attn[1]]

            table_ASD = np.array(
                [left_gaze_x_list_ASD, left_gaze_y_list_ASD, right_gaze_x_list_ASD, right_gaze_y_list_ASD,
                 touch_x_list_ASD,
                 touch_y_list_ASD, touch_hard_list_ASD, duration_ASD])
            df_ASD = pd.DataFrame(table_ASD)

            if not os.path.exists(f'results/visualization/ASD{user_id}_gamelevel_{level}_attn'):
                os.mkdir(f'results/visualization/ASD{user_id}_gamelevel_{level}_attn')
            df_ASD.to_csv(f'results/visualization/ASD{user_id}_gamelevel_{level}_attn/attention_{idx}.csv', index=None,
                         header=None)

    # if label == 1:
    #
    #     left_gaze_x_list_ASD = np.concatenate(left_gaze_x_list_ASD,axis=0)
    #     left_gaze_y_list_ASD = np.concatenate(left_gaze_y_list_ASD,axis=0)
    #     right_gaze_x_list_ASD = np.concatenate(right_gaze_x_list_ASD,axis=0)
    #     right_gaze_y_list_ASD = np.concatenate(right_gaze_y_list_ASD,axis=0)
    #     touch_x_list_ASD = np.concatenate(touch_x_list_ASD,axis=0)
    #     touch_y_list_ASD = np.concatenate(touch_y_list_ASD,axis=0)
    #     touch_hard_list_ASD = np.concatenate(touch_hard_list_ASD)
    #     duration_ASD = np.concatenate(duration_ASD)

        whole_table_ASD = np.array(
            [whole_left_gaze_x_list_ASD, whole_left_gaze_y_list_ASD, whole_right_gaze_x_list_ASD, whole_right_gaze_y_list_ASD, whole_touch_x_list_ASD,
             whole_touch_y_list_ASD, whole_touch_hard_list_ASD, whole_duration_ASD])
        whole_df_ASD = pd.DataFrame(whole_table_ASD)
        if not os.path.exists(f'results/visualization/origin'):
            os.mkdir(f'results/visualization/origin')
        whole_df_ASD.to_csv(f'results/visualization/origin/ASD{user_id}_gamelevel_{level}_origin.csv', index=None,
                      header=None)


        whole_table_TD = np.array(
            [whole_left_gaze_x_list_TD, whole_left_gaze_y_list_TD, whole_right_gaze_x_list_TD, whole_right_gaze_y_list_TD, whole_touch_x_list_TD,
             whole_touch_y_list_TD, whole_touch_hard_list_TD, whole_duration_TD])
        whole_df_TD = pd.DataFrame(whole_table_TD)
        whole_df_TD.to_csv(f'results/visualization/origin/TD{user_id}_gamelevel_{level}_origin.csv', index=None, header=None)


