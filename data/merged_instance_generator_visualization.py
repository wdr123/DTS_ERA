import os
import pandas as pd
import numpy as np
import torch
from utilFiles.set_deterministic import make_deterministic
np.set_printoptions(suppress=True)

class ASDTDTaskGenerator(object):
    def __init__(self, setname, data_path = "Final_Dataset", args = None,  batch_size = 2):
        if not args:
            print('Pass args')
            raise NotImplementedError

        make_deterministic(seed=0)

        self.task_counter = 0
        seed = args.seed

        self.args = args

        data_path = data_path + str(seed)


        self._data_path = os.path.join(os.getcwd(), "data")
        self._data_path = os.path.join(self._data_path, data_path)

        # Train or test
        path = os.path.join(self._data_path, setname)

        self.ind_to_path_merged = {}

        self.path_to_ind_merged = {}

        self.path_merged_data_merged = {}

        self.path_merged_length_both = {}


        self.ind_to_av_lev = {}
        self.ind_to_class_level = {}


        # self.num_window = 8
        self.setname = setname

        #Player id
        count = 0
        players_list = sorted(os.listdir(path)) #In train or test the users [p01, p02,...]
        print(players_list)
        self.num_asd_user_levls = 0
        for index, player_name in enumerate(players_list): #[P01, P02,...]
            player_path = os.path.join(path, player_name)
            player_episode_id = sorted([int(x.split('merged')[1].split('.csv')[0]) for x in os.listdir(player_path) if 'merged' in x])
            # print(player_episode_id)
            if 'ASD' in player_name:
                self.num_asd_user_levls += len(player_episode_id)

            for index2, data in enumerate(player_episode_id):
                merged_episode_path = os.path.join(player_path, 'merged'+str(data)+'.csv')

                self.ind_to_class_level[count] = ('ASD' in player_name)*1
                self.ind_to_path_merged[count] = merged_episode_path
                self.path_to_ind_merged[merged_episode_path] = count

                count += 1

                self.path_merged_data_merged[merged_episode_path] = self.get_data_from_path(merged_episode_path) #All user-level type data

                self.path_merged_length_both[merged_episode_path] = self.get_length_from_path(merged_episode_path)

        self.data_len = len(self.path_merged_length_both)
        # self.episode_len = []
        # for k,v in self.path_merged_length_both.items():
        #     self.episode_len.append(v[1])
        self.num_window = 100
        if self.setname == 'train':
            self.already_visit_user = []

        # SAVE DICT TEST INDEX TO FILE
        save_df = pd.DataFrame.from_dict(self.ind_to_path_merged, orient="index")
        save_df.to_csv(f"{setname}_ind_to_path_merged.csv")

        # print(os.listdir(path))



    def get_data_from_path(self, episode_path):
        data = pd.read_csv(episode_path).to_numpy()[:,1:]
        return data

    def get_length_from_path(self, a):
        l1_start = 0
        l1_end = len(pd.read_csv(a).to_numpy())
        return [l1_start, l1_end]

    def __len__(self):
        if self.setname in [ 'test', 'val']:
            return self.data_len
        return 150 #self.data_len

    def __getitem__(self, idx):
        #Randomly select a player and then the corresponding episode
        while True:
            if self.task_counter % 2 == 0:
                user_ep_id = np.random.randint(self.num_asd_user_levls)
            else:
                user_ep_id = self.num_asd_user_levls + np.random.randint(self.data_len - self.num_asd_user_levls)
            if user_ep_id not in self.already_visit_user:
                break
        # user_ep_id = np.random.randint(self.data_len)
        # print("self: ", self.data_len, self.num_asd_user_levls, user_ep_id, self.task_counter)
        self.already_visit_user.append(user_ep_id)

        if self.setname in ['test', 'val']:
            user_ep_id = self.task_counter % self.data_len
        self.task_counter += 1

        # The path of the selected player-episode
        merged_path = self.ind_to_path_merged[user_ep_id]
        game_level = merged_path.split('merged')[1].split('.csv')[0]
        user_id = (merged_path.split('\\')[-2])[-2:]
        # print(user_id)

        # Data to begin

        ranges = self.path_merged_length_both[merged_path]
        pos_begin = np.random.randint(ranges[0], ranges[1]-self.num_window)
        pos_end = pos_begin + self.num_window

        # Corresponding data
        merged_data = self.path_merged_data_merged[merged_path]

        windowed_ = False
        if windowed_:
            merged_data = merged_data[pos_begin:pos_end]


        merged_data = torch.tensor(merged_data)
        # All
        if self.setname in ['test', 'val']: # and not windowed_:
            merged_data = torch.tensor(self.path_merged_data_merged[merged_path])


        class_label = torch.tensor([self.ind_to_class_level[user_ep_id]])

        # gaze_start = 0
        # if self.args.is_lstm: #No time feature
        gaze_start = 0
        touch_data = merged_data[:,-3:]
        # else:
        #     touch_data = torch.cat((merged_data[:,:1],merged_data[:,-3:]),1)
        # gaze_data = merged_data[:,1:-3]
        gaze_data = merged_data[:,gaze_start:-5] #Don't use pd
        touch_gaze_data = torch.cat((gaze_data, touch_data), 1)
        # gaze_data = merged_data[:,gaze_start:-7] #-5
        # gaze_data = torch.cat((merged_data[:,:1],merged_data[:,-7:-5]),1) #-5
        # gaze_data = merged_data[:,-7:-5]
        # gaze_data = torch.cat((merged_data[:,-7:-6],merged_data[:,-8:-7]),1) #-5

        return touch_gaze_data, class_label, game_level, user_id
