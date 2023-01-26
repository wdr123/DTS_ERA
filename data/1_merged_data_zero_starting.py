import os
import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)

formatted_data_path = "formatted_data"

source_file_name = "merged_data"
if not os.path.exists(source_file_name):
    os.mkdir(source_file_name)


def get_seconds(x):
    x = x.split(':')
    secs = int(x[1])*60 + float(x[2])
    return secs

window_size = 0.1 # A hyperparameter


def get_windowed_np_array(the_np_array):
    final_array = []
    start_time, count = 0.0, 0
    at_pos = 0.0

    running_average = [0 for _ in the_np_array[0]]

    for t in the_np_array:
        # Running average and count
        count += 1
        running_average = [x + y for x, y in zip(running_average, t)]

        # How much time has passed
        current_time = t[0]
        time_difference = current_time - start_time

        if time_difference > window_size:
            running_average = [x / count for x in running_average]
            count = 0

            start_time = current_time


            while at_pos < current_time: at_pos += window_size
            running_average[0] = at_pos

            final_array.append(running_average)
            running_average = [0 for _ in running_average]

    return np.array(final_array)





def main():
    #The users
    entries = os.listdir(formatted_data_path)
    entries.sort()
    print(entries)

    #For each user
    for e in entries:
        user_path = os.path.join(formatted_data_path, e)

        #Destination
        user_folder_name = os.path.join(source_file_name, e)
        if not os.path.exists(user_folder_name):
            os.mkdir(user_folder_name)

        print("user path: ", user_path)
        user_paths = os.listdir(user_path)

        #Consider only the gaze
        path = [int(x.split('gaze')[1].split('.csv')[0]) for x in user_paths if 'gaze' in x] #Integer indicating the data levels
        for each in path:
            each_gaze = 'gaze' + str(each) + '.csv'
            each_touch = 'touch' + str(each) + '.csv'
            each_merged = "merged" + str(each) + ".csv"
            print("Touch: ", each_touch, "Gaze: ", each_gaze)

            if each in [11, 12] and user_path == 'formatted_data/' + 'ASD-P05' or \
               each in [1, 2, 3] and user_path == 'formatted_data/' + 'ASD-P06' or \
               each in [1,2,3,4,7] and user_path == 'formatted_data/' + 'ASD-P08':
                #ASD User combination to skip
                continue

            if each in [2] and user_path == 'formatted_data/' + 'TD-P03' or \
               each in [1, 2, 3, 4, 5] and user_path == 'formatted_data/' + 'TD-P04' or \
               each in [1, 2, 3, 4, 5, 6, 7, 8, 9] and user_path == 'formatted_data/' + 'TD-P06' or \
               each in [1, 2, 3] and user_path == 'formatted_data/' + 'TD-P09' or \
               each in [10, 11] and user_path == 'formatted_data/' + 'TD-P10' or \
               each in [1, 2, 3, 4] and user_path == 'formatted_data/' + 'TD-P11' or \
               each in [10, 11] and user_path == 'formatted_data/' + 'TD-P12':
                    #TD User combination to skip
                continue


            touch_dest_path = os.path.join(user_folder_name, each_touch)
            gaze_dest_path = os.path.join(user_folder_name, each_gaze)
            merged_dest_path = os.path.join(user_folder_name, each_merged)

            each_gaze_path = os.path.join(user_path, each_gaze)
            gaze_df = pd.read_csv(each_gaze_path, sep=',')

            time_0_gaze = get_seconds(gaze_df['Time'].tolist()[0])

            gaze_df['Time'] = gaze_df['Time'].apply(lambda x: get_seconds(x)) #Change Time to seconds
            # print(gaze_df['Time'].tolist()[:10])

            each_touch_path = os.path.join(user_path, each_touch)
            touch_df = pd.read_csv(each_touch_path, sep=',')
            # print("touch df: ", touch_df.columns)

            time_0_touch = get_seconds(touch_df['Time'].tolist()[0])
            touch_df['Time'] = touch_df['Time'].apply(lambda x: get_seconds(x)) #Change time to seconds

            min_time = min(time_0_touch, time_0_gaze) #First Tracked time should be 0

            gaze_df['Time'] = gaze_df['Time'].apply(lambda x: x - min_time) #Make Corresponding change to time

            touch_df['Time'] = touch_df['Time'].apply(lambda x: x - min_time)
            # print(gaze_df['Time'].tolist()[:10])
            # print(touch_df['Time'].tolist()[:10])

            touch_np = touch_df.to_numpy()
            gaze_np = gaze_df.to_numpy()

            touch_np_array = get_windowed_np_array(touch_np)
            gaze_np_array = get_windowed_np_array(gaze_np)

            touch_pd_df = pd.DataFrame(touch_np_array, columns=touch_df.columns)
            gaze_pd_df = pd.DataFrame(gaze_np_array, columns=gaze_df.columns)

            merged_df = pd.merge(gaze_pd_df,touch_pd_df)


            # print(touch_pd_df)
            # print(touch_np_array[:, :3])
            # print(gaze_np_array[:, :3])

            # print(gaze_pd_df)

            merged_df.to_csv(merged_dest_path)
            # gaze_pd_df.to_csv(gaze_dest_path)


if __name__== "__main__":
    main()
