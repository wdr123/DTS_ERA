###
'''
To do: Given the Raw data, combine the data
'''
###

import os

#The folder after removing the necessary ones
raw_dataset_path = 'raw'
target_file_name = "formatted_data"


def break_level_gaze_csv(filepath, dest_path, identifier='g'):

    #Write code to read the csv file, save the data level wise i.e. level 1: level1 all data to save in a file

    # print(filepath, "identifier: ", identifier)
    lines = []
    with open(filepath) as f:
        lines = f.readlines()
    num_levels = 0

    #Handling the Data Structure (Consider if there are 12 levels, otherwise ignore)
    for l in lines:
        if "level" in l.lower():
            num_levels += 1

    if num_levels < 10:
        print(" Filepath: ", filepath, "Not 12 levels, # levels: ", num_levels)
        return

    new_level = False
    data_start = False

    for l in lines:
        if "level" in l.lower():
            dir_name = os.path.join(dest_path, identifier+str(int(l[6:]))+'.csv')
            new_level = True
            # print("dir_name: ", dir_name)

        elif new_level and "time" in l.lower():
            #Mannually set the header
            # header = ['Time', 'Right Gaze Point:X','Right Gaze Point:Y', 'Left Gaze Point:X', 'Left Gaze Point:Y', 'Right Pupil Diamter','Left Pupil Diameter\n']
            header = 'Time,RGPX,RGPY,LGPX,LGPY,RPD,LPD \n'
            # print("header: ", header)

            data_start = True
            new_level = False

#             if not os.path.exists(dir_name):
#                 f = open(dir_name, "w")
#                 f.write(header)
#                 f.close()

            f = open(dir_name, "w")
            f.write(header)
            f.close()

        elif data_start and 'nan' not in l and "**************************" not in l:
            f = open(dir_name, "a")
            l = ",".join(l.split()[1:]) + "\n" #Don't need date, only time

            f.write(l)
            f.close()

        elif data_start and '******************************************************************' in l:
            data_start = False

    return 0

def break_level_touch_csv(filepath, dest_path, identifier='p'):
    # Write code to read the csv file, save the data level wise i.e. level 1: level1 all data to save in a file
    print(filepath, "identifier: ", identifier)
    lines = []
    with open(filepath) as f:
        lines = f.readlines()
    num_levels = 0

    # Handling the Data Structure (Consider if there are 12 levels, otherwise ignore)
    for l in lines:
        if "level" in l.lower():
            num_levels += 1

    if num_levels < 10:
        print(" Filepath: ", filepath, "Not 12 levels, # levels: ", num_levels)
        return
    else:
        print("Num levels: ", num_levels)

    new_level = False
    data_start = False

    for l in lines:
        if "level" in l.lower():
            dir_name = os.path.join(dest_path, identifier + str(int(l[6:8]))+ '.csv')
            new_level = True
            print("dir_name: ", dir_name)

        elif new_level and "time" in l.lower():
            # Mannually set the header

            data_start = True
            new_level = False

            header = "Time,x,y,z\n"

#             if not os.path.exists(dir_name):
#                 f = open(dir_name, "w")
#                 f.write(header)
#                 f.close()

            f = open(dir_name, "w")
            f.write(header)
            f.close()

        elif data_start and 'nan' not in l and '****' not in l:
            f = open(dir_name, "a")
            l = ",".join(l.split()) + "\n"
            f.write(l)
            f.close()

        elif data_start and '*****************************************' in l:
            data_start = False
    return 0

def save_to_file(user_id, path, target_files):

    if 'AD-0A' in user_id: user_id = 'ASD-' + user_id[5:]
    if not os.path.exists(target_file_name):
        os.mkdir(target_file_name)

    user_folder_name = os.path.join(target_file_name, user_id)
    if not os.path.exists(user_folder_name):
        os.mkdir(user_folder_name)

    for tf in target_files:
        if "gaze" in tf.lower():
            print("gaze")
            the_gaze_path = os.path.join(path, tf)
            break_level_gaze_csv(the_gaze_path, user_folder_name, "gaze")
            #Read the Gaze data level by level
            #Save Gaze data To the File level by level. Sth like g1, g2, ...
        elif "path" in tf.lower():
            print("touch")
            the_touch_path = os.path.join(path, tf)
            break_level_touch_csv(the_touch_path, user_folder_name, "touch")
            # Read the path data level by level
            # Save Gaze data To the File level by level. Sth like p1, p2, ...
        else:
            continue




def main():
    #All the users
    entries = os.listdir(raw_dataset_path)
    entries.sort()

    #See user names
    print("User names: ", entries)

    # Each User
    for e in entries:
        user_path = os.path.join(raw_dataset_path, e) #raw/TD1
        for dir in os.listdir(user_path): #Data
            user_data_path = os.path.join(user_path, dir) #raw/TD1/Data
            touch_gaze_path = os.listdir(user_data_path)
            if 'recordGaze.txt' not in touch_gaze_path: #Another sub folder- Go one more level
                for fol in touch_gaze_path:
                    act_user_data_path = os.path.join(user_data_path, fol)

                    if 'AD-0AP05' == e and fol == '12' or \
                        'AD-0AP06' == e and fol == '4-12' or \
                        'AD-0AP08' == e and fol == '1-4' :
                        continue
                    if 'TD-P04' == e and fol == 'A' or \
                        'TD-P06' == e and fol in ['1', '1-9'] or \
                        'TD-P09' == e and fol == '1-3' or \
                        'TD-P10' == e and fol == '1-11' or \
                        'TD-P11' == e and fol == 'TD_P11' or \
                        'TD-P12' == e and fol == 'TD_P12A' :
                        continue

                    act_touch_gaze_path = os.listdir(act_user_data_path)
                    if 'recordGaze.txt' not in act_touch_gaze_path:
                        print(e, dir, act_touch_gaze_path)
                        print("Not correct")
                        raise FileNotFoundError
                    else:
                        # print(e, act_user_data_path, act_touch_gaze_path)
                        save_to_file(e, act_user_data_path, act_touch_gaze_path)
                        # I have logposition, recordgaze and record touch
            else:
                # print("dir: ", e, user_data_path, touch_gaze_path)
                save_to_file(e, user_data_path, touch_gaze_path)
                # I have logposition, recordgaze and record touch

if __name__ == "__main__":
    main()

