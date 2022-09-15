from __future__ import print_function

import os
import torch.utils.data
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
from torch.distributions.normal import Normal
from data.merged_instance_generator_visualization import ASDTDTaskGenerator
from utilFiles.the_args import get_args
from utilFiles.set_deterministic import make_deterministic




assert torch.cuda.is_available()
args, _ = get_args()
args.seed = 0
args.std = 0.1
args.model = "attention_only"
attn = True
T = 5
obs_window = 10
num_attn = 2
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
print(device)

make_deterministic(seed=args.seed)
# kwargs = {'num_workers': 1, 'pin_memory': True} if device.type=='cuda' else {}

from models.RAM_ASD_TD import MODEL, LOSS, adjust_learning_rate

#Data Loaders
train_ds = ASDTDTaskGenerator("train", data_path="Fine_grain", args = args)
# train_dl_bs16 = DataLoader(train_ds,batch_size=16,shuffle=True)
train_dl = DataLoader(train_ds,batch_size=1,shuffle=False)

model = MODEL(args).to(device)
loss_fn = LOSS(T=args.T, gamma=args.gamma, device=device).to(device)


model.load_state_dict(torch.load(f'results/checkpoint/RAM_3_train_test_latent256_attention_only_combine_selen10_msize2_time_step5_sd0.pth'))
loss_fn.load_state_dict(torch.load(f'results/checkpoint/LOSS_3_train_test_latent256_attention_only_combine_selen10_msize2_time_step5_sd0.pth'))

import csv
def save_to_csv(args, all_dicts, iter=0):
    header, values = [], []
    for d in all_dicts:
        for k, v in d.items():
            header.append(k)
            values.append(v)

    if not os.path.exists(f'./results/all/test'):
        os.makedirs(f'./results/all/test')
    save_model_name = f"Save_test3000_latent256_attention_only_combine_selen10_msize2_time_step5_sd0.csv"
    save_model_path = os.path.join(f'./results/all/test', save_model_name)

    # save_model_name = "Debug.csv"
    if iter == 0:
        with open(save_model_path, 'w+') as f:
            writer_obj = csv.writer(f)
            writer_obj.writerow(header)
    with open(save_model_path, 'a') as f:
        writer_obj = csv.writer(f)
        writer_obj.writerow(values)


'''
Evaluation
'''

model.eval()
train_aloss, train_lloss, train_bloss, train_reward = 0, 0, 0, 0
count, count_asd, count_td, num_correct, num_correct_asd, num_correct_td = 0, 0, 0, 0, 0, 0

for batch_idx, (touch_gaze_data, label, level, user_id) in enumerate(train_dl):
    attn_start = []

    touch_gaze_data_back = touch_gaze_data.detach().clone()
    touch_gaze_data = touch_gaze_data[:,:,1:].to(device).float()
    touch_data = touch_gaze_data[:,:,4:]
    gaze_data = touch_gaze_data[:,:,:4]
    label = label.to(device).float()
    model.initialize(touch_data.size(0), device, std=0.1)
    loss_fn.initialize(touch_data.size(0))
    for t in range(args.T):
        logpi, action = model(touch_data, gaze_data)
        loc = model.show_loc()
        aloss, lloss, bloss, reward = loss_fn(action, label, logpi)
        attn_start.append(loc.cpu().numpy())

    attn_start = np.array(attn_start).flatten()

    label = label[0]
    level = level[0]
    touch_gaze_data = touch_gaze_data_back[0].cpu().numpy()
    time_lapse = touch_gaze_data[:,0]
    time_lapse_before = np.concatenate([np.array([0]),touch_gaze_data[:-1,0]],axis=0)
    touch_gaze_data = touch_gaze_data[:,1:]
    touch_gaze_data[touch_gaze_data <= 0] = 0.01
    touch_gaze_data_notime = np.minimum(touch_gaze_data[:,:-1], 1)
    touch_gaze_data[:,:-1] = touch_gaze_data_notime
    touch_gaze_data[:,-1] = touch_gaze_data[:,-1] / np.linalg.norm(touch_gaze_data[:,-1])

    attn_start = (attn_start * len(touch_gaze_data)).astype(np.int)
    # attn_start = np.random.randint(low=len(touch_gaze_data) - obs_window, high=None, size=num_attn*T)
    attn_end = np.minimum(attn_start + obs_window, len(touch_gaze_data))

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
            if not os.path.exists(f'results/visualization/only_attention_data/TD{user_id}_gamelevel_{level}_attn'):
                os.mkdir(f'results/visualization/only_attention_data/TD{user_id}_gamelevel_{level}_attn')
            df_TD.to_csv(f'results/visualization/only_attention_data/TD{user_id}_gamelevel_{level}_attn/attention_{idx}.csv', index=None,
                         header=None)

        whole_table_TD = np.array(
            [whole_left_gaze_x_list_TD, whole_left_gaze_y_list_TD, whole_right_gaze_x_list_TD,
             whole_right_gaze_y_list_TD, whole_touch_x_list_TD,
             whole_touch_y_list_TD, whole_touch_hard_list_TD, whole_duration_TD])
        whole_df_TD = pd.DataFrame(whole_table_TD)
        whole_df_TD.to_csv(f'results/visualization/only_attention_data/origin/TD{user_id}_gamelevel_{level}_origin.csv', index=None,
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

            if not os.path.exists(f'results/visualization/only_attention_data/ASD{user_id}_gamelevel_{level}_attn'):
                os.mkdir(f'results/visualization/only_attention_data/ASD{user_id}_gamelevel_{level}_attn')
            df_ASD.to_csv(f'results/visualization/only_attention_data/ASD{user_id}_gamelevel_{level}_attn/attention_{idx}.csv', index=None,
                         header=None)

        whole_table_ASD = np.array(
            [whole_left_gaze_x_list_ASD, whole_left_gaze_y_list_ASD, whole_right_gaze_x_list_ASD, whole_right_gaze_y_list_ASD, whole_touch_x_list_ASD,
             whole_touch_y_list_ASD, whole_touch_hard_list_ASD, whole_duration_ASD])
        whole_df_ASD = pd.DataFrame(whole_table_ASD)
        if not os.path.exists(f'results/visualization/only_attention_data/origin'):
            os.mkdir(f'results/visualization/only_attention_data/origin')
        whole_df_ASD.to_csv(f'results/visualization/only_attention_data/origin/ASD{user_id}_gamelevel_{level}_origin.csv', index=None,
                      header=None)


