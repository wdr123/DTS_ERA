'''
Modified from construct_bags.py
'''
import os
import numpy as np
import random
import pandas as pd
import argparse
from distutils.dir_util import copy_tree
import shutil

def get_seed():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type = int, default=18,help="Seed for the code")
    args = parser.parse_args()
    return args.seed

def copy_dataset(folder_name, source_path, dest_path, pos_usernames, neg_usernames):
    dest_name = os.path.join(dest_path, folder_name)
    os.mkdir(dest_name)

    usernames = []
    usernames += [x for x in pos_usernames]
    usernames += [x for x in neg_usernames]

    for usr in usernames:
        src_pth = os.path.join(source_path, usr)
        dst_pth = os.path.join(dest_name, usr)
        copy_tree(src_pth, dst_pth)

if __name__ == "__main__":
    rand_seed = get_seed()
    print(rand_seed)
    source_name = "merged_data"
    random.seed(rand_seed)

    user_names = sorted(os.listdir(source_name))
    print('user names: ', user_names)

    #From construct_bags.py
    pos_usernames, neg_usernames = [], []
    n_train = 9
    n_val = 0
    n_test = 2
    for user_name in user_names:
        if 'TD' in user_name:
            neg_usernames.append(user_name)
        elif 'ASD' in user_name:
            pos_usernames.append(user_name)

    pos_usernames = np.array(pos_usernames)
    neg_usernames = np.array(neg_usernames)
    random.shuffle(pos_usernames)
    random.shuffle(neg_usernames)
    pos_train_usernames = pos_usernames[:n_train]
    neg_train_usernames = neg_usernames[:n_train]
    pos_val_usernames = pos_usernames[n_train: n_train + n_val]
    neg_val_usernames = neg_usernames[n_train: n_train + n_val]
    pos_test_usernames = pos_usernames[n_train + n_val:]
    neg_test_usernames = neg_usernames[n_train + n_val:]

    #Now, reorganize as required
    source_path = os.path.join(os.getcwd(), source_name)

    print("pos tr usr: ", pos_train_usernames)

    ds_name = "Fnl_Ds"+str(rand_seed)
    dest_path = os.path.join(os.getcwd(),ds_name)

    try:
        shutil.rmtree(dest_path)
    except:
        print("No existing dataset")
    os.mkdir(dest_path)

    #Train
    print("Now reorganize the users")
    copy_dataset("train", source_path, dest_path, pos_train_usernames, neg_train_usernames)
    print("Train done")
    copy_dataset("test", source_path, dest_path, pos_test_usernames, neg_test_usernames)
    print("Test done")
    # copy_dataset("val", source_path, dest_path, pos_val_usernames, neg_val_usernames)
    # print("Val Done")
    print("Done")
