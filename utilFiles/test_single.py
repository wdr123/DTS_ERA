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
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

make_deterministic(seed=args.seed)
kwargs = {'num_workers': 1, 'pin_memory': True} if device.type=='cuda' else {}

from models.RAM_ASD_TD_partial import MODEL, LOSS, adjust_learning_rate

#Data Loaders
test_ds = ASDTDTaskGenerator("test", data_path="Fine_grain", args = args)
test_dl = DataLoader(test_ds,batch_size=1,shuffle=True, **kwargs)



# T = 5
# lr = 0.0001
# std = 0.25
# scale = 1
# decay = 0.5
model = torch.load(f'../results/checkpoint/RAM_partial_{args.identifier}_latent{args.latent}_{args.model}_{args.attention}_selen{args.selen}_msize{args.msize}_time_step{args.T}_sd{args.seed}.pth')
loss_fn = torch.load(f'../results/checkpoint/LOSS_partial_{args.identifier}_latent{args.latent}_{args.model}_{args.attention}_selen{args.selen}_msize{args.msize}_time_step{args.T}_sd{args.seed}.pth')
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
    # model.initialize(touch_data.size(0), device)
    # loss_fn.initialize(touch_data.size(0))

    for _ in range(args.T):
        logpi, action = model(touch_data, gaze_data)
        aloss, lloss, bloss, reward = loss_fn(action, label, logpi)
    if args.model == "no_attention":
        loss = aloss
    else:
        loss = aloss+lloss+bloss
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


print('====> Average loss: a {:.4f} l {:.4f} b {:.4f} Reward: {:.1f} Test Accuracy:{:.1f} Test ASD Accuracy:{:.1f} Test TD Accuracy:{:.1f}'.format(
      test_aloss / len(test_dl.dataset),
      test_lloss / len(test_dl.dataset),
      test_bloss / len(test_dl.dataset),
      test_reward *100/ len(test_dl.dataset),
))