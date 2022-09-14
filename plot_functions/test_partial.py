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

T = 5
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
print(device)


# kwargs = {'num_workers': 1, 'pin_memory': True} if device.type=='cuda' else {}

from models.RAM_ASD_TD import MODEL, LOSS, adjust_learning_rate



import csv
def save_to_csv(label, usr_id, level, partial, seed, all_dicts, iter=0):
    header, values = [], []
    for d in all_dicts:
        for k, v in d.items():
            header.append(k)
            values.append(v)
    if label == 0:
        if not os.path.exists(f'./results/all/partial/TD/{partial}'):
            os.makedirs(f'./results/all/partial/TD/{partial}')
        save_model_name = f"Partial{partial}_test2000_user{usr_id}_gl{level}_sd{seed}.csv"
        save_model_path = os.path.join(f'./results/all/partial/TD/{partial}', save_model_name)
    else:
        if not os.path.exists(f'./results/all/partial/ASD/{partial}'):
            os.makedirs(f'./results/all/partial/ASD/{partial}')
        save_model_name = f"Partial{partial}_test2000_user{usr_id}_gl{level}_sd{seed}.csv"
        save_model_path = os.path.join(f'./results/all/partial/ASD/{partial}', save_model_name)

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

for seed in range(0,10,2):
    args.seed = seed
    make_deterministic(seed=args.seed)
    args.std = 0.1
    # Data Loaders
    test_ds = ASDTDTaskGenerator("test", data_path="Fine_grain", args=args)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)

    model = MODEL(args).to(device)
    loss_fn = LOSS(T=args.T, gamma=args.gamma, device=device).to(device)
    model.load_state_dict(torch.load(f'results/checkpoint/RAM_2_partial_latent256_combine_combine_selen10_msize2_time_step5_sd{seed}.pth'))
    loss_fn.load_state_dict(torch.load(f'results/checkpoint/LOSS_2_partial_latent256_combine_combine_selen10_msize2_time_step5_sd{seed}.pth'))


    model.eval()

    for batch_idx, (touch_gaze_data, label, level, user_id) in enumerate(test_dl):
        for partial in range(2,11,2):
            partial = partial / 10
            num_instance = touch_gaze_data.size(1)
            touch_gaze_data_back = touch_gaze_data[:, :int(partial*num_instance), 1:].to(device).float()
            touch_data = touch_gaze_data_back[:, :int(partial*num_instance), 4:]
            gaze_data = touch_gaze_data_back[:, :int(partial*num_instance), :4]
            label = label.to(device).float()
            model.initialize(touch_data.size(0), device, std=0.1)
            loss_fn.initialize(touch_data.size(0))
            for t in range(args.T):
                logpi, action = model(touch_data, gaze_data)
                aloss, lloss, bloss, reward = loss_fn(action, label, logpi)
            if args.model == "no_attention":
                loss = aloss
            else:
                loss = aloss + lloss + bloss
            # aloss.item() += aloss.item()
            # lloss.item() += lloss.item()
            # bloss.item() += bloss.item()
            # test_reward += reward.item()

            pred = 1 * (torch.sigmoid(action.detach()) > 0.5)
            # print("pred: ", pred)
            # print("label: ", label)
            num_correct = (1.0 * (label.squeeze() == pred.squeeze())).item()

            num_correct_asd = (1.0 * (label.squeeze() == 1.0) * (pred.squeeze() == 1.0)).item()
            num_correct_td = (1.0 * (label.squeeze() == 0.0) * (pred.squeeze() == 0.0)).item()

            print('====> Average loss: a {:.4f} l {:.4f} b {:.4f} Reward: {:.1f}'.format(
                aloss.item(),
                lloss.item(),
                bloss.item(),
                reward.item()))

            test_dict = {
                'Partial': partial,
                'Test Accuracy (%)': reward.item() * 100,
                'Test Action Loss': aloss.item(),
                'Test Location Loss': lloss.item(),
                'Test Baseline Loss': bloss.item(),
                'If Correct': num_correct,
                'If ASD Correct': num_correct_asd,
                'If TD Correct': num_correct_td,
            }

            user_id = user_id[0]
            level = level[0]
            save_to_csv(label, user_id, level, partial, seed, [test_dict], 0)