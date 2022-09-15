import os
import numpy as np
import pandas as pd

dir = '../data/Fine_grain18'


def get_length_from_path(a):
    l1_start = 0
    l1_end = len(pd.read_csv(a).to_numpy())
    return [l1_start, l1_end]

test_min = 10000
test_max = -1
total_avg = 0
total_count = 0
avg_asd = 0
count_asd = 0
avg_td = 0
count_td = 0

for train_test_split in os.listdir(dir):
    train_test_split_path = os.path.join(dir, train_test_split)
    for user_folder in os.listdir(train_test_split_path):
        user_path = os.path.join(train_test_split_path, user_folder)

        if 'ASD' in user_folder:
            for gl_file in os.listdir(user_path):
                gl_path = os.path.join(user_path, gl_file)
                length = get_length_from_path(gl_path)
                count_asd += 1
                total_count += 1
                avg_asd += length[1]
                total_avg += length[1]
                test = length[1]
                print('{}:{}:'.format(user_folder, gl_file), length[1])
                if test < test_min:
                    test_min = test
                if test >= test_max:
                    test_max = test
        elif 'TD' in user_folder:
            for gl_file in os.listdir(user_path):
                gl_path = os.path.join(user_path, gl_file)
                length = get_length_from_path(gl_path)
                count_td += 1
                total_count += 1
                avg_td += length[1]
                total_avg += length[1]
                test = length[1]
                print('{}:{}:'.format(user_folder, gl_file), length[1])
                if test < test_min:
                    test_min = test
                if test >= test_max:
                    test_max = test

print('ASD_avg_length_{}:'.format(avg_asd/count_asd))
print('TD_avg_length_{}:'.format(avg_td/count_td))

print('total_acg', total_avg/total_count)
print('avg_min', test_min)
print('avg_max', test_max)