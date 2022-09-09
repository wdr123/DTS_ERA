import numpy as np
from matplotlib import pyplot as plt
import math
import matplotlib
font = {'family' : 'normal',
        'size'   : 20}

matplotlib.rc('font', **font)
plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['savefig.dpi'] = 100

# Centerness Count
y1= [245, 432, 328, 412, 357, 825, 639, 457, 206] # TD
y2= [283, 467, 354, 378, 517, 448, 209] # ASD

index = np.arange(5)
bar_width = 0.2

plt.bar(index, y1, color='b', width=bar_width, label='TD')
plt.bar(index+bar_width, y2, color='g', width=bar_width, label='ASD')
plt.xticks(index+bar_width, ('1','2','3','4','5'))
# plt.xticks(index+bar_width, ('romance','drama','comedy','thriller','others'))


plt.xlabel('Number of Pattern')
plt.ylabel('Count')
plt.legend(loc='upper right', ncol=3, labelspacing=0.1, handletextpad=0.1, fontsize=10)
plt.tight_layout()
plt.savefig('Center_Count.png')
plt.show()
plt.close()