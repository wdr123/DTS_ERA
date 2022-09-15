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


model.load_state_dict(torch.load(f'results/checkpoint/RAM_2_partial_latent256_combine_combine_selen10_msize2_time_step5_sd0.pth'))
loss_fn.load_state_dict(torch.load(f'results/checkpoint/LOSS_2_partial_latent256_combine_combine_selen10_msize2_time_step5_sd0.pth'))

import csv
def save_to_csv(args, all_dicts, iter=0):
    header, values = [], []
    for d in all_dicts:
        for k, v in d.items():
            header.append(k)
            values.append(v)

    if not os.path.exists(f'./results/all/test'):
        os.makedirs(f'./results/all/test')
    save_model_name = f"Save_test3000_latent256_combine_combine_selen10_msize2_time_step5_sd0.csv"
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

deep_set_asd = []
deep_set_td = []


for batch_idx, (touch_gaze_data, label, level, user_id) in enumerate(train_dl):

    touch_gaze_data_back = touch_gaze_data.detach().clone()
    touch_gaze_data = touch_gaze_data[:,:,1:].to(device).float()
    touch_data = touch_gaze_data[:,:,4:]
    gaze_data = touch_gaze_data[:,:,:4]
    label = label.to(device).float()
    model.initialize(touch_data.size(0), device, std=0.1)
    loss_fn.initialize(touch_data.size(0))
    for t in range(args.T):
        logpi, action = model(touch_data, gaze_data)
        aloss, lloss, bloss, reward = loss_fn(action, label, logpi)

    label = label[0]
    level = level[0]

    if t==args.T-1:
        deep_set = model.show_deep_set().detach().cpu().numpy()
        if label == 1:
            deep_set_asd.append(deep_set)
        else:
            deep_set_td.append(deep_set)

# print(len(deep_set_asd))
# print(len(deep_set_td))
# print(deep_set_asd[0].shape)
# print(deep_set_td[0].shape)
td_form = np.concatenate(deep_set_td, axis=0)
asd_form = np.concatenate(deep_set_asd, axis=0)

np.savetxt('results/visualization/result_deepset/ASD.csv', asd_form, delimiter=',' )
np.savetxt('results/visualization/result_deepset/TD.csv', td_form, delimiter=',')