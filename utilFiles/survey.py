import os
import numpy as np
import pandas as pd

dir = '../data/Fine_grain18'


def get_length_from_path(a):
    l1_start = 0
    l1_end = len(pd.read_csv(a).to_numpy())
    return [l1_start, l1_end]

avg_min = 10000
avg_max = -1

for split in os.listdir(dir):
    split_path = os.path.join(dir, split)
    for user in os.listdir(split_path):
        user_path = os.path.join(split_path, user)
        # avg = 0
        for merged in os.listdir(user_path):
            fnl_path = os.path.join(user_path, merged)
            length = get_length_from_path(fnl_path)
            avg = length[1]
            print('{}:{}:'.format(user, merged), length[1])
            if avg < avg_min:
                avg_min = avg
            if avg >= avg_max:
                avg_max = avg

        print('{}_length:'.format(user), avg)

    print('avg_min', avg_min)
    print('avg_max', avg_max)