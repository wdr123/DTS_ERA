import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from data.merged_instance_generator_visualization import ASDTDTaskGenerator
from torch.utils.data import DataLoader
from utilFiles.the_args import get_seed
from utilFiles.set_deterministic import make_deterministic

args, _ = get_seed()
T = 5
obs_window = 10
num_attn = 10
test_dl = DataLoader(ASDTDTaskGenerator("test", data_path="dataset", args = args),batch_size=1,shuffle=True)
make_deterministic(seed=0)
x = []
y = []
x2 = []
y2 = []
x3 = []
y3 = []
z3 = []
feature_repo = []
pca_2_x = []
pca_2_y = []
pca_3_x = []
pca_3_y = []
pca_3_z = []
ind_to_label = []
ind_to_level = []
label_to_ind = {}
level_to_ind = {}



for batch_idx, (touch_gaze_data, label, level) in enumerate(test_dl):

    label = label[0]
    level = level[0]
    touch_gaze_data = touch_gaze_data[0].cpu().numpy()
    touch_gaze_data[touch_gaze_data <= 0] = 0.01
    touch_gaze_data_before = np.minimum(touch_gaze_data[:,:-1], 1)
    touch_gaze_data[:,:-1] = touch_gaze_data_before
    touch_gaze_data[:,-1] = touch_gaze_data[:,-1] / np.linalg.norm(touch_gaze_data[:,-1])

    left_gaze_x_list = touch_gaze_data[:, 0]
    left_gaze_y_list = touch_gaze_data[:, 1]
    right_gaze_x_list = touch_gaze_data[:, 2]
    right_gaze_y_list = touch_gaze_data[:, 3]
    touch_x_list = touch_gaze_data[:, 4]
    touch_y_list = touch_gaze_data[:, 5]
    touch_hard_list = touch_gaze_data[:, 6]
    duration = 0.1 * np.ones(len(left_gaze_x_list))
    duration_copy = np.copy(duration)
    touch_hard_list_copy = np.copy(touch_hard_list)

    # duration = 0.1 * np.ones(obs_window)

    if label == 1:
        attn_start = np.random.randint(low=len(touch_gaze_data)-obs_window,high=None,size=num_attn)
        attn_end = attn_start + obs_window

        for attn in list(zip(attn_start, attn_end)):
            duration_copy[attn[0]:attn[1]] = duration[attn[0]:attn[1]] * num_attn * 100
            touch_hard_list_copy[attn[0]:attn[1]] = touch_hard_list[attn[0]:attn[1]] * num_attn * 100

            # left_gaze_x_list = touch_gaze_data[attn[0]:attn[1], 0]
            # left_gaze_y_list = touch_gaze_data[attn[0]:attn[1], 1]
            # right_gaze_x_list = touch_gaze_data[attn[0]:attn[1], 2]
            # right_gaze_y_list = touch_gaze_data[attn[0]:attn[1], 3]
            # touch_x_list = touch_gaze_data[attn[0]:attn[1], 4]
            # touch_y_list = touch_gaze_data[attn[0]:attn[1], 5]
            # touch_hard_list_copy = touch_gaze_data[attn[0]:attn[1], 6] * num_attn
            # duration_copy = duration * num_attn


    # else:
    #     for attn in list(zip(attn_start, attn_end)):
    #         left_gaze_x_list = touch_gaze_data[attn[0]:attn[1], 0]
    #         left_gaze_y_list = touch_gaze_data[attn[0]:attn[1], 1]
    #         right_gaze_x_list = touch_gaze_data[attn[0]:attn[1], 2]
    #         right_gaze_y_list = touch_gaze_data[attn[0]:attn[1], 3]
    #         touch_x_list = touch_gaze_data[attn[0]:attn[1], 4]
    #         touch_y_list = touch_gaze_data[attn[0]:attn[1], 5]
    #         touch_hard_list_copy = touch_gaze_data[attn[0]:attn[1], 6]
    #         duration_copy = duration



    table = np.array([left_gaze_x_list,left_gaze_y_list,right_gaze_x_list,right_gaze_y_list,touch_x_list,touch_y_list,touch_hard_list_copy, duration_copy])
    df = pd.DataFrame(table)

    label = "ASD" if label==1 else "TD"
    df.to_csv(f'results/visualization/test_{batch_idx}.csv',index=None,header=None)
    # df.to_csv(f'results/visualization/attention_{batch_idx}.csv',index=None,header=None)