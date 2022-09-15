import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Use CUDA if available, otherwise error
assert torch.cuda.is_available()
device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

from data.merged_instance_generator_bs import ASDTDTaskGenerator


import random
import numpy as np
import torch

from utilFiles.the_args import get_seed
from utilFiles.set_deterministic import make_deterministic

args, _ = get_seed()
make_deterministic(seed=args.seed)

#Data Loaders
bs = 8
train_dl = DataLoader(ASDTDTaskGenerator("train", data_path="dataset", args = args),batch_size=bs,shuffle=True)
# val_dl = DataLoader(ASDTDTaskGenerator("val", seed = args.seed),batch_size=1,shuffle=True)
test_dl = DataLoader(ASDTDTaskGenerator("test", data_path="dataset", args = args),batch_size=1,shuffle=True)

if args.weighted:
    if args.is_lstm:
        from models.ASDvsTDModel_wmean import ASD_TD_CNP_LSTM as model
        print("LSTM model")
    else:
        from models.ASDvsTDModel_wmean import ASD_TD_CNP as model
else:
    if args.is_lstm:
        from models.ASDvsTDModel_mean import ASD_TD_CNP_LSTM as model
        print("LSTM model")
    else:
        from models.ASDvsTDModel_mean import ASD_TD_CNP as model


model = model(is_gaze=args.is_gaze, is_touch=args.is_touch).to(device)

criterion = nn.BCEWithLogitsLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

print(model)

import csv
def save_to_csv(all_dicts,iter=0):
    header, values = [], []
    for d in all_dicts:
        for k, v in d.items():
            header.append(k)
            values.append(v)

    save_model_name = f"Save_{args.identifier}_Batch8_sd{args.seed}_t_{args.is_touch}_g_{args.is_gaze}_lstm_{args.is_lstm}.csv"
    # save_model_name = "Debug.csv"
    if iter == 0:
        with open(save_model_name, 'a') as f:
            writer_obj = csv.writer(f)
            writer_obj.writerow(header)
    with open(save_model_name, 'a') as f:
        writer_obj = csv.writer(f)
        writer_obj.writerow(values)

def test(identifier):
    if identifier == "test":
        the_dataloader = test_dl
    elif identifier == "val":
        print("NO validation now")
        raise NotImplementedError
        # the_dataloader = val_dl
    with torch.no_grad():
        model.eval()
        accuracy, count,loss = 0, 0, 0.0
        accuracy_1 = 0
        acc_asd, count_asd, acc_td, count_td = 0,0,0,0
        for d in the_dataloader:
            model.eval()
            model.zero_grad()
            optimizer.zero_grad()

            touch_data, gaze_data, label = d
            label = label.to(device)
            touch_data = touch_data.to(device).float()
            gaze_data = gaze_data.to(device).float()
            # add_data = add_data.to(device).float()

            pred = model(touch_data, gaze_data, add_data=0)

            # print("label: ", label)
            # print("prediction: ", pred)
            loss += criterion(pred, label.float())

            pred = 1*(torch.sigmoid(pred)>0.5)
            # print("pred: ", pred)
            # print("label: ", label)
            count += 1
            accuracy += 1.0*(label.squeeze() == pred.squeeze())
            accuracy_1 +=1.0*(label.squeeze() == 1.0)

            acc_asd += 1.0 * (label.squeeze() == 1.0) *(pred.squeeze()==1.0)
            acc_td += 1.0 * (label.squeeze() == 0.0) *(pred.squeeze()==0.0)
            count_asd += 1.0 * (label.squeeze() == 1.0)
            count_td += 1.0 * (label.squeeze() == 0.0)

    av_acc = accuracy/count
    av_acc = av_acc.detach().cpu().numpy().item()
    av_loss = loss/count
    av_loss = av_loss.detach().cpu().numpy().item()

    av_acc_asd = acc_asd/count_asd
    av_acc_td = acc_td/count_td
    av_acc_asd = av_acc_asd.detach().cpu().numpy().item()
    av_acc_td = av_acc_td.detach().cpu().numpy().item()
    av_acc_both = (av_acc_td+av_acc_asd)/2

    print("{} acc: ".format(identifier), accuracy/count, "count: ", count, "all: ", accuracy_1/count, 'loss: ', av_loss)

    the_dict = {
        identifier + " loss":av_loss,
        identifier + " ASD acc": av_acc_asd,
        identifier + " TD acc": av_acc_td,
        identifier + " AVG acc": av_acc_both,
        identifier + " acc": av_acc,
    }
    print('the_dict: ', the_dict)

    return the_dict

def one_iteration_training(touch_data, gaze_data, label):
    model.zero_grad()
    optimizer.zero_grad()
    model.train()

    # print('touch: ', touch_data.shape)

    if touch_data.shape[1] <=5 or gaze_data.shape[1] <= 5:
        print("Not enough data")
        raise NotImplementedError

    pred = model(touch_data,gaze_data, add_data=0)
    # print("pred: ", pred)
    # print("label: ", label)
    loss = criterion(pred,label.float())
    # print("loss: ", loss)
    loss.backward()
    optimizer.step()
    pred = 1 * (torch.sigmoid(pred) > 0.5)
    # print("pred: ", pred)
    # print("label: ", label)
    accuracy = torch.mean(1.0 * (label.squeeze() == pred.squeeze()))
    return loss.detach().cpu().numpy(), accuracy



def main():
    for tr_it in range(1000):
        test_dict = test("test")
        # val_dict = test("val")

        loss_tr, count = 0.0, 0
        accuracy = 0.0
        overall_count, fail_count = 0, 0
        for d in train_dl:
            overall_count += 1
            touch_data, gaze_data, label = d
            label = label.to(device)
            touch_data = touch_data.to(device).float()
            gaze_data = gaze_data.to(device).float()
            # add_data = add_data.to(device).float()
            if touch_data.shape[1] <= 5 or gaze_data.shape[1] <= 5:
                # print("Not enough data")
                fail_count += 1
                continue

            count += 1
            model.zero_grad()
            optimizer.zero_grad()
            model.train()

            loss, acc = one_iteration_training(touch_data, gaze_data, label)
            loss_tr += loss
            accuracy += acc
        print("Fail count, overall count", fail_count, overall_count)
        print("Tr Loss at it: ", tr_it, " loss: ", loss_tr / count, "accuracy: ", accuracy / count)

        tr_loss = loss_tr / count
        tr_loss = tr_loss.item()
        tr_acc = accuracy / count
        tr_acc = tr_acc.detach().cpu().numpy().item()

        train_dict = {
            'Train acc': tr_acc,
            'Train loss':tr_loss
        }

        all_dicts_list = [train_dict,test_dict]

        save_to_csv(all_dicts_list,tr_it)


if __name__ == "__main__":
    main()
