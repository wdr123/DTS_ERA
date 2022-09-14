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

y1 = [91, 98, 86, 78, 198, 45, 55, 64, 35] # TD multi-modal
y2 = [66, 94, 81, 91, 218, 72, 42, 55, 31] # ASD multi-modal


# y1= [74, 72, 55, 42, 49, 56, 58, 26, 28, 32, 25, 52, 34, 65, 28, 54] # TD gaze
# y2= [92, 81, 67, 68, 52, 72, 63, 34, 39, 22, 14, 38, 15, 48, 13, 32] # ASD gaze

touch_gaze_label = ('Strip', 'Oval', 'Circle', 'BCS', 'BSS', 'ICS', 'ISS', 'Triangle', 'Line', 'ST','SL', 'LS', 'GS', 'LL', 'GL', 'Hook')
multi_modal_label = ('LL','LC','LS','CL','CC','CS','SL','SC','SS')

index = np.arange(len(y1))
bar_width = 0.2

plt.bar(index, y1, color='b', width=bar_width, label='TD')
plt.bar(index+bar_width, y2, color='g', width=bar_width, label='ASD')
# plt.xticks(index+bar_width, touch_gaze_label)
plt.xticks(index+bar_width, multi_modal_label)


plt.xlabel('Number of Pattern')
plt.ylabel('Count')
plt.legend(loc='upper right', ncol=3, labelspacing=0.1, handletextpad=0.1, fontsize=10)
plt.tight_layout()
# plt.savefig('../results/visualization/statistical_count/Left_Gaze_Pattern_Count.png')
# plt.savefig('../results/visualization/statistical_count/Touch_Hard_Pattern_Count.png')
plt.savefig('../results/visualization/statistical_count/Multi_Modal_Pattern_Count.png')
plt.close()