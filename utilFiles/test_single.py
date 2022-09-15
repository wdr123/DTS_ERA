from __future__ import print_function

import os

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import numpy as np
from torch.distributions.normal import Normal
from data.merged_instance_generator_partial import ASDTDTaskGenerator
from utilFiles.the_args import get_args
from utilFiles.set_deterministic import make_deterministic




assert torch.cuda.is_available()
args, _ = get_args()
args.seed = 0
args.std = 0.1
args.model = "attention_only"
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
print(device)

make_deterministic(seed=args.seed)
# kwargs = {'num_workers': 1, 'pin_memory': True} if device.type=='cuda' else {}

from models.RAM_ASD_TD import MODEL, LOSS, adjust_learning_rate

#Data Loaders
test_ds = ASDTDTaskGenerator("test", data_path="Fine_grain", args = args)
test_dl = DataLoader(test_ds,batch_size=1,shuffle=True)

model = MODEL(args).to(device)
loss_fn = LOSS(T=args.T, gamma=args.gamma, device=device).to(device)


# model.load_state_dict(torch.load(f'results/checkpoint/RAM_2_partial_latent256_combine_combine_selen10_msize2_time_step5_sd0.pth'))
# loss_fn.load_state_dict(torch.load(f'results/checkpoint/LOSS_2_partial_latent256_combine_combine_selen10_msize2_time_step5_sd0.pth'))
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
    save_model_name = f"Save_test3000_latent256_only_attention_combine_selen10_msize2_time_step5_sd0.csv"
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
test_aloss, test_lloss, test_bloss, test_reward = 0, 0, 0, 0
count, count_asd, count_td, num_correct, num_correct_asd, num_correct_td = 0, 0, 0, 0, 0, 0

for batch_idx, (touch_data, gaze_data, label) in enumerate(test_dl):
    touch_data = touch_data.to(device).float()
    gaze_data = gaze_data.to(device).float()
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
    test_aloss += aloss.item()
    test_lloss += lloss.item()
    test_bloss += bloss.item()
    test_reward += reward.item()

    pred = 1 * (torch.sigmoid(action.detach()) > 0.5)
    # print("pred: ", pred)
    # print("label: ", label)
    count += 1
    num_correct += (1.0 * (label.squeeze() == pred.squeeze())).item()

    num_correct_asd += (1.0 * (label.squeeze() == 1.0) * (pred.squeeze() == 1.0)).item()
    num_correct_td += (1.0 * (label.squeeze() == 0.0) * (pred.squeeze() == 0.0)).item()
    count_asd += (1.0 * (label.squeeze() == 1.0)).item()
    count_td += (1.0 * (label.squeeze() == 0.0)).item()

print('====> Average loss: a {:.4f} l {:.4f} b {:.4f} Reward: {:.1f}'.format(
    test_aloss / len(test_dl.dataset),
           test_lloss / len(test_dl.dataset),
           test_bloss / len(test_dl.dataset),
           test_reward * 100 / len(test_dl.dataset)))

test_dict = {
    'Test Accuracy (%)': test_reward * 100 / len(test_dl.dataset),
    'Test Action Loss': test_aloss / len(test_dl.dataset),
    'Test Location Loss': test_lloss / len(test_dl.dataset),
    'Test Baseline Loss': test_bloss / len(test_dl.dataset),
    'Test Accuracy': num_correct / count,
    'Test ASD Accuracy': num_correct_asd / count_asd,
    'Test TD Accuracy': num_correct_td / count_td,
}

all_dicts_list = [test_dict]

save_to_csv(args, all_dicts_list, 0)