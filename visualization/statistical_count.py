import numpy as np
from matplotlib import pyplot as plt
import math
import matplotlib
font = {'family' : 'normal',
        'size'   : 20}

matplotlib.rc('font', **font)
plt.rcParams['figure.figsize'] = (16, 16)
plt.rcParams['savefig.dpi'] = 100

# Centerness Count
# y1 = [76, 66, 57, 48, 44, 51, 47, 28, 33, 37, 27, 33, 21, 70, 53, 59] # TD touch
# y2 = [84, 77, 63, 62, 51, 68, 68, 37, 46, 18, 16, 28, 13, 52, 35, 32] # ASD touch

# y1 = [91, 98, 86, 78, 198, 45, 55, 64, 35] # TD multi-modal
# y2 = [66, 94, 81, 91, 218, 72, 42, 55, 31] # ASD multi-modal

# y1 = [50,58,31,28] # TD shape
# y2 = [30,32,56,52] # ASD shape
# multi_modal_shape_label = np.array(['Oval-LL','LL-LL', 'Strip-ISS','ICS-ISS',])

# clutter_idx = [0,1,2,3,5,7,]
# local_idx = [4,6,8,9,10,11,13,15]
# global_idx = [12,14]
# area_label = np.array(['Clutter','Local', 'Global',])

# y1= np.array([74, 72, 55, 42, 49, 56, 58, 26, 28, 32, 25, 52, 34, 65, 28, 54]) # TD gaze
# y2= np.array([92, 81, 67, 68, 52, 72, 63, 34, 39, 22, 14, 38, 15, 48, 13, 32]) # ASD gaze

# m1 = [sum(y1[clutter_idx]),sum(y1[local_idx]),sum(y1[global_idx])]
# m2 = [sum(y2[clutter_idx]),sum(y2[local_idx]),sum(y2[global_idx])]

# touch_gaze_label = np.array(['Strip', 'Oval', 'Circle', 'BCS', 'BSS', 'ICS', 'ISS', 'Triangle', 'Line', 'ST','SL', 'LS', 'GS', 'LL', 'GL', 'Hook'])
# multi_modal_label = np.array(['LL','LC','LS','CL','CC','CS','SL','SC','SS'])

y1= np.array([186, 156, 32, 76, 145, 76, 78, 74, 77]) # Multimodal Area TD
y2= np.array([212, 184, 36, 78, 132, 72, 63, 64, 59]) # Multimodal Area ASD
area_label = np.array(['CC','CL', 'CG', 'LC', 'LL', 'LG', \
'GC','GL','GG'])

# double_pattern_idx = [3,4,5,6]
# tri_pattern_idx = [7,8,9,10]
# many_pattern_idx = [11,12,]

index = np.arange(len(y1))
bar_width = 0.2

plt.bar(index, y1, color='b', width=bar_width, label='TD')
plt.bar(index+bar_width, y2, color='g', width=bar_width, label='ASD')
# plt.xticks(index+bar_width, touch_gaze_label)
plt.xticks(index+bar_width, area_label)


plt.xlabel('Category of Pattern')
plt.ylabel('Count')
plt.legend(loc='upper right', ncol=3, labelspacing=0.1, handletextpad=0.1, fontsize=10)
plt.tight_layout()
# plt.savefig('../results/visualization/statistical_count/Left_Gaze_double_Pattern_Count.png')
# plt.savefig('../results/visualization/statistical_count/Touch_Hard_Pattern_Count.png')
plt.savefig('../results/visualization/statistical_count/Area_Pattern_Mutimoal_Count.png')
plt.close()