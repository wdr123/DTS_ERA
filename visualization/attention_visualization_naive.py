import numpy as np
import matplotlib.pyplot as plt
from data.merged_instance_generator_visualization import ASDTDTaskGenerator
from torch.utils.data import DataLoader
from utilFiles.the_args import get_args

args, _ = get_args()
T = 5
test_dl = DataLoader(ASDTDTaskGenerator("test", data_path="dataset", args = args),batch_size=1,shuffle=True)
x = []
y = []
x2 = []
y2 = []
x3 = []
y3 = []
z3 = []
feature_repo = []
pca_2_x = []
pca_2_y = []
pca_3_x = []
pca_3_y = []
pca_3_z = []
ind_to_label = []
ind_to_level = []
label_to_ind = {}
level_to_ind = {}


for batch_idx, (touch_gaze_data, label, level) in enumerate(test_dl):
    # N = 10
    # x = np.random.rand(N)
    # y = np.random.rand(N)
    # x2 = np.random.rand(N)
    # y2 = np.random.rand(N)
    # x3 = np.random.rand(N)
    # y3 = np.random.rand(N)
    # area = np.random.rand(N) * 1000
    # fig = plt.figure()
    # ax = plt.subplot()
    label = label[0]
    level = level[0]
    touch_gaze_data = touch_gaze_data[0].cpu().numpy()
    x.append( touch_gaze_data[:, 0] ) # Gaze left
    y.append( touch_gaze_data[:, 1] )
    x2.append( touch_gaze_data[:, 2] ) # Gaze right
    y2.append( touch_gaze_data[:, 3] )
    x3.append( touch_gaze_data[:, 4] ) # Touch
    y3.append( touch_gaze_data[:, 5] )
    z3.append( touch_gaze_data[:, 6] )
    ind_to_label.append(label)
    ind_to_level.append(level)

    if label in label_to_ind.keys():
        label_to_ind[label].append(batch_idx)
    else:
        label_to_ind["ASD" if label==1 else "TD"] = [batch_idx]
    if level in level_to_ind.keys():
        level_to_ind[level].append(batch_idx)
    else:
        level_to_ind[level] = [batch_idx]
    # ax.scatter(x, y, s=area, alpha=0.5)
    # ax.scatter(x2, y2, s=area, c='green', alpha=0.6)
    # ax.scatter(x3, y3, s=area, c=area, marker='v', cmap='Reds', alpha=0.7)  # 更换标记样式，另一种颜色的样式
    # plt.show()


feature_repo.append([x,y])
feature_repo.append([x2,y2])
feature_repo.append([x3,y3])
feature_repo.append([x3,y3,z3])
feature_repo.append([pca_2_x,pca_2_y])
feature_repo.append([pca_3_x,pca_3_y,pca_3_z])
feature_ind_to_meaning = {}
feature_ind_to_meaning[0] = 'left_gaze'
feature_ind_to_meaning[1] = 'right_gaze'
feature_ind_to_meaning[2] = 'touch_2d'
feature_ind_to_meaning[3] = 'touch_3d'
feature_ind_to_meaning[4] = 'pca_2d'
feature_ind_to_meaning[5] = 'pca_4d'

print(level_to_ind)


for lv in level_to_ind.keys():
    flag_pd = False
    flag_asd = False
    if len(level_to_ind[lv])>=2:
        # print('lv:',lv)
        pd_user_lv = []
        asd_user_lv = []
        for idx in level_to_ind[lv]:

            if ind_to_label[idx] == 1:
                # print('1',idx)
                asd_user_lv.append(idx)
                flag_asd = True
            else:
                # print('0',idx)
                pd_user_lv.append(idx)
                flag_pd = True

        if (not flag_asd) or (not flag_pd):
            continue

        for feature_select in range(6):
            feature = feature_repo[feature_select]
            # asd_x_feature_set = []
            # asd_y_feature_set = []
            # pd_x_feature_set = []
            # pd_y_feature_set = []
            # for id in asd_user_lv:
            #     for asd_x_feature in feature[0][id]:
            #         asd_x_feature = list(asd_x_feature)
            #         asd_x_feature_set.extend(asd_x_feature)
            #     for asd_y_feature in feature[1][asd_user_lv]:
            #         asd_y_feature = list(asd_y_feature)
            #         asd_y_feature_set.extend(asd_y_feature)
            # for pd_x_feature in feature[0][pd_user_lv]:
            #     pd_x_feature = list(pd_x_feature)
            #     pd_x_feature_set.extend(pd_x_feature)
            # for pd_y_feature in feature[1][pd_user_lv]:
            #     pd_y_feature = list(pd_y_feature)
            #     pd_y_feature_set.extend(pd_y_feature)
            if (feature_select == 3) or (feature_select == 4) or (feature_select == 5):
                # fig = plt.figure()
                # ax = plt.subplot()
                # for idx in level_to_ind[lv]:
                #     x = feature[0][idx]
                #     y = feature[1]
                #     z = feature[2]
                continue

            else:
                fig = plt.figure(figsize=(12,10))
                ax = plt.subplot(1,3,1)
                # ax_attn = plt.subplot(1,3,2)
                # ax_correct = plt.subplot(1,3,3)
                for idx in level_to_ind[lv]:
                    # print('idx', idx)
                    x = feature[0][idx]
                    y = feature[1][idx]
                    label = "ASD" if ind_to_label[idx]==1 else "TD"
                    # area = list(np.ones(len(x)))

                    x_mean = []
                    y_mean = []
                    area_mean = []
                    # x_mean.extend(x)
                    # y_mean.extend(y)
                    # area_mean.extend(area)
                    x_mean.append(np.mean(x))
                    y_mean.append(np.mean(y))
                    area_mean.append(len(x))
                    ax.scatter(x_mean, y_mean, s=area_mean, alpha=0.5, label=f'{label}')
                    # ax_attn.scatter(x_mean, y_mean, s=area_mean, alpha=0.5, label=f'{label}')
                    # ax_correct.scatter(x_mean, y_mean, s=area_mean, alpha=0.5, label=f'{label}')

                    # x_attn = []
                    # x_attn.extend([x[np.random.randint(len(x))] for _ in range(T)])
                    # x_attn.append(np.mean(x_attn))
                    # y_attn = []
                    # # y_attn.extend([y[np.random.randint(len(y))] for _ in range(T)])
                    # y_attn.append(np.mean(y_attn))
                    # area_attn = []
                    # # area_attn.extend([len(x) / T for _ in range(T)])
                    # area_attn.append(len(x))
                    # ax_attn.scatter(x_attn, y_attn, s=area_attn, c='green', alpha=0.6, label=f'{label}')
                    # ax_correct.scatter(x_attn, y_attn, s=area_attn, c='green', alpha=0.6, label=f'{label}')
                    #
                    # x_correct = []
                    # y_correct = []
                    # area_correct = []
                    # x_correct.append(np.mean([x_attn[-1], x_mean[-1]]))
                    # y_correct.append(np.mean([y_attn[-1], y_mean[-1]]))
                    # area_correct.append(len(x) * 2)
                    # ax_correct.scatter(x_correct, y_correct, s=area_correct, marker='v',
                    #            alpha=0.7, label=f'{label}')  # 更换标记样式，另一种颜色的样式
                    #
                    ax.set_title('mean')
                    # ax_attn.set_title('attention')
                    # ax_correct.set_title('correction')

                plt.legend()
                plt.savefig(f'./visualization/gamelevel_{lv}_{feature_ind_to_meaning[feature_select]}.png')
                plt.close()
