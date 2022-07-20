from __future__ import print_function
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import numpy as np
from torch.distributions.normal import Normal
from data.merged_instance_generator_mean import ASDTDTaskGenerator
from utilFiles.the_args import get_seed
from utilFiles.set_deterministic import make_deterministic




assert torch.cuda.is_available()
# batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args, _ = get_seed()
make_deterministic(seed=args.seed)
kwargs = {'num_workers': 1, 'pin_memory': True} if device.type=='cuda' else {}
# train_dl = torch.utils.data.DataLoader(datasets.MNIST('../data', train=True,
#                                            transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])),
#                                            batch_size=batch_size, shuffle=True, **kwargs)
# test_dl = torch.utils.data.DataLoader(datasets.MNIST('../data', train=False,
#                                            transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])),
#                                            batch_size=batch_size, shuffle=True, **kwargs)

if args.weighted:
    from models.RAM_ASD_TD_wmean import MODEL, LOSS
    print("weighted")
else:
    from models.RAM_ASD_TD_mean import MODEL, LOSS

#Data Loaders
bs = 16
train_dl = DataLoader(ASDTDTaskGenerator("train", data_path="dataset", args = args),batch_size=bs,shuffle=True)
# val_dl = DataLoader(ASDTDTaskGenerator("val", seed = args.seed),batch_size=1,shuffle=True)
test_dl = DataLoader(ASDTDTaskGenerator("test", data_path="dataset", args = args),batch_size=1,shuffle=True)



T = 5
lr = 0.0001
std = 0.25
# scale = 1
decay = 0.95
model = MODEL(window_size= 50, std = std).to(device)
loss_fn = LOSS(T=T, gamma=1, device=device).to(device)
# optimizer = optim.Adam(list(model.parameters())+list(loss_fn.parameters()), lr=lr)
optimizer = optim.Adam(list(model.parameters())+list(loss_fn.parameters()), lr=lr,weight_decay=1e-5)
print(model)

import csv
def save_to_csv(args, all_dicts, iter=0):
    header, values = [], []
    for d in all_dicts:
        for k, v in d.items():
            header.append(k)
            values.append(v)


    save_model_name = f"Save_{args.identifier}_Batch16_sd{args.seed}.csv"
    # save_model_name = "Debug.csv"
    if iter == 0:
        with open(save_model_name, 'w+') as f:
            writer_obj = csv.writer(f)
            writer_obj.writerow(header)
    with open(save_model_name, 'a') as f:
        writer_obj = csv.writer(f)
        writer_obj.writerow(values)



for epoch in range(1000):
    '''
    Training
    '''
    # adjust_learning_rate(optimizer, epoch, lr, decay)
    model.train()
    train_aloss, train_lloss, train_bloss, train_reward = 0, 0, 0, 0

    for batch_idx, (touch_data, gaze_data, label) in enumerate(train_dl):
        touch_data = touch_data.to(device).float()
        gaze_data = gaze_data.to(device).float()
        label = label.to(device).float()
        optimizer.zero_grad()
        model.initialize(touch_data.size(0), device)
        loss_fn.initialize(touch_data.size(0))
        for _ in range(T):
            logpi, action = model(touch_data, gaze_data)
            aloss, lloss, bloss, reward = loss_fn(action, label, logpi)  # loss_fn stores logpi during intermediate time-stamps and returns loss in the last time-stamp
        loss = aloss+lloss+bloss  
        loss.backward()
        optimizer.step()
        train_aloss += aloss.item()
        train_lloss += lloss.item()
        train_bloss += bloss.item()
        train_reward += reward.item()


    print('====> Epoch: {} Average loss: a {:.4f} l {:.4f} b {:.4f} Reward: {:.1f}'.format(
          epoch, train_aloss / len(train_dl.dataset),
          train_lloss / len(train_dl.dataset), 
          train_bloss / len(train_dl.dataset),
          train_reward *100/ len(train_dl.dataset)))

    train_dict = {
        'Epoch' : epoch,
        'Train acc': train_reward *100 / len(train_dl.dataset),
        'Train Action Loss': train_aloss / len(train_dl.dataset),
        'Train Location Loss': train_lloss / len(train_dl.dataset),
        'Train Baseline Loss': train_bloss / len(train_dl.dataset),
    }

    # save_to_csv("train", train_dict, epoch)
    # uncomment below line to save the model
    # torch.save([model.state_dict(), loss_fn.state_dict(), optimizer.state_dict()],'results/final'+str(epoch)+'.pth')

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
        model.initialize(touch_data.size(0), device)
        loss_fn.initialize(touch_data.size(0))
        for _ in range(T):
            logpi, action = model(touch_data, gaze_data)
            aloss, lloss, bloss, reward = loss_fn(action, label, logpi)
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


    print('====> Epoch: {} Average loss: a {:.4f} l {:.4f} b {:.4f} Reward: {:.1f}'.format(
          epoch, test_aloss / len(test_dl.dataset),
          test_lloss / len(test_dl.dataset), 
          test_bloss / len(test_dl.dataset),
          test_reward *100/ len(test_dl.dataset)))


    test_dict = {
        'Epoch': epoch,
        'Test Accuracy (%)': test_reward * 100 / len(test_dl.dataset),
        'Test Action Loss': test_aloss / len(test_dl.dataset),
        'Test Location Loss': test_lloss / len(test_dl.dataset),
        'Test Baseline Loss': test_bloss / len(test_dl.dataset),
        'Test Accuracy': num_correct / count,
        'Test ASD Accuracy': num_correct_asd / count_asd,
        'Test TD Accuracy': num_correct_td / count_td,
    }


    all_dicts_list = [train_dict, test_dict]

    save_to_csv(args, all_dicts_list, epoch)


