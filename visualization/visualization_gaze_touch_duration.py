import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from data.merged_instance_generator_visualization import ASDTDTaskGenerator
from torch.utils.data import DataLoader
from utilFiles.the_args import get_args
from utilFiles.set_deterministic import make_deterministic

args, _ = get_args()
attn = True
T = 5
obs_window = 100
num_attn = 2
train_dl = DataLoader(ASDTDTaskGenerator("train", data_path="Fine_grain", args = args),batch_size=1,shuffle=True)
make_deterministic(seed=0)




for batch_idx, (touch_gaze_data, label, level, user_id) in enumerate(train_dl):

    left_gaze_x_list_ASD = []
    left_gaze_y_list_ASD = []
    right_gaze_x_list_ASD = []
    right_gaze_y_list_ASD = []
    touch_x_list_ASD = []
    touch_y_list_ASD = []
    touch_hard_list_ASD = []
    duration_ASD = []

    left_gaze_x_list_TD = []
    left_gaze_y_list_TD = []
    right_gaze_x_list_TD = []
    right_gaze_y_list_TD = []
    touch_x_list_TD = []
    touch_y_list_TD = []
    touch_hard_list_TD = []
    duration_TD = []

    label = label[0]
    level = level[0]
    touch_gaze_data = touch_gaze_data[0].cpu().numpy()
    touch_gaze_data[touch_gaze_data <= 0] = 0.01
    touch_gaze_data_before = np.minimum(touch_gaze_data[:,:-1], 1)
    touch_gaze_data[:,:-1] = touch_gaze_data_before
    touch_gaze_data[:,-1] = touch_gaze_data[:,-1] / np.linalg.norm(touch_gaze_data[:,-1])

    # attn_start = np.random.randint(low=len(touch_gaze_data) - obs_window, high=None, size=num_attn)
    # attn_end = attn_start + obs_window

    if label==0:
        left_gaze_x_list_TD.append(touch_gaze_data[:, 0])
        left_gaze_y_list_TD.append(touch_gaze_data[:, 1])
        right_gaze_x_list_TD.append(touch_gaze_data[:, 2])
        right_gaze_y_list_TD.append(touch_gaze_data[:, 3])
        touch_x_list_TD.append(touch_gaze_data[:, 4])
        touch_y_list_TD.append(touch_gaze_data[:, 5])
        touch_hard_list_TD.append(touch_gaze_data[:, 6] )
        duration_TD.append(0.1*np.ones(len(touch_gaze_data)))

        left_gaze_x_list_TD = np.concatenate(left_gaze_x_list_TD, axis=0)
        left_gaze_y_list_TD = np.concatenate(left_gaze_y_list_TD, axis=0)
        right_gaze_x_list_TD = np.concatenate(right_gaze_x_list_TD, axis=0)
        right_gaze_y_list_TD = np.concatenate(right_gaze_y_list_TD, axis=0)
        touch_x_list_TD = np.concatenate(touch_x_list_TD, axis=0)
        touch_y_list_TD = np.concatenate(touch_y_list_TD, axis=0)
        touch_hard_list_TD = np.concatenate(touch_hard_list_TD, axis=0)
        duration_TD = np.concatenate(duration_TD, axis=0)

        table_TD = np.array(
            [left_gaze_x_list_TD, left_gaze_y_list_TD, right_gaze_x_list_TD, right_gaze_y_list_TD, touch_x_list_TD,
             touch_y_list_TD, touch_hard_list_TD, duration_TD])
        df_TD = pd.DataFrame(table_TD)
        df_TD.to_csv(f'results/visualization/origin_new/TD{user_id}_gamelevel_{level}_origin.csv', index=None, header=None)

    else:
        left_gaze_x_list_ASD.append(touch_gaze_data[:, 0])
        left_gaze_y_list_ASD.append(touch_gaze_data[:, 1])
        right_gaze_x_list_ASD.append(touch_gaze_data[:, 2])
        right_gaze_y_list_ASD.append(touch_gaze_data[:, 3])
        touch_x_list_ASD.append(touch_gaze_data[:, 4])
        touch_y_list_ASD.append(touch_gaze_data[:, 5])
        touch_hard_list_ASD.append(touch_gaze_data[:, 6] )
        duration_ASD.append(0.1 * np.ones(len(touch_gaze_data)))

        left_gaze_x_list_ASD = np.concatenate(left_gaze_y_list_ASD, axis=0)
        left_gaze_y_list_ASD = np.concatenate(left_gaze_y_list_ASD, axis=0)
        right_gaze_x_list_ASD = np.concatenate(right_gaze_x_list_ASD, axis=0)
        right_gaze_y_list_ASD = np.concatenate(right_gaze_y_list_ASD, axis=0)
        touch_x_list_ASD = np.concatenate(touch_x_list_ASD, axis=0)
        touch_y_list_ASD = np.concatenate(touch_y_list_ASD, axis=0)
        touch_hard_list_ASD = np.concatenate(touch_hard_list_ASD, axis=0)
        duration_ASD = np.concatenate(duration_ASD, axis=0)

        table_ASD = np.array(
            [left_gaze_x_list_ASD, left_gaze_y_list_ASD, right_gaze_x_list_ASD, right_gaze_y_list_ASD, touch_x_list_ASD,
             touch_y_list_ASD, touch_hard_list_ASD, duration_ASD])
        df_ASD = pd.DataFrame(table_ASD)
        df_ASD.to_csv(f'results/visualization/origin_new/ASD{user_id}_gamelevel_{level}_origin.csv', index=None, header=None)












