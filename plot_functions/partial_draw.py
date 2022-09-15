import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import random
import seaborn as sns

np.random.seed(0)
random.seed(0)

plt.rcParams.update({'font.size': 15})
matplotlib.rcParams['legend.fontsize'] = 15

clrs = sns.color_palette("husl", 2)

fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(111)
ax.grid()

# set labels and tick font size
ax.set_xlabel('Episode Trajectory Completion Degree', fontsize = 15)
ax.set_ylabel('ASD Test Accuracy', fontsize = 15)
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)

xasd = np.array([0.2,0.4,0.6,0.8,1])
yasd = np.array([0.8, 0.843, 0.875, 0.894, 0.932])
error = np.array([0.04, 0.043, 0.053, 0.052,  0.051]) # ASD_std

# yasd = [89.29,89.97,90.14,91.12,91.37]
# target_name = ['ASD 20%','ASD 40%','ASD 60%','ASD 80%','ASD 100%',]
# error = np.random.normal(loc=0,scale=0.1,size=5)


ax.plot(xasd, yasd, label='ASD', c=clrs[0])
ax.fill_between(xasd, yasd - error, yasd + error, alpha=0.3, facecolor=clrs[0])

plt.title("Test results on partial completion trajectory for ASD user")
plt.savefig('../results/visualization/partial/partial_asd.png')
plt.close()

fig = plt.figure(figsize=(8,5))
ax1 = fig.add_subplot(111)
ax1.grid()

# set labels and tick font size
ax1.set_xlabel('Episode Trajectory Completion Degree', fontsize = 15)
ax1.set_ylabel('TD Test Accuracy', fontsize = 15)
ax1.tick_params(axis='x', labelsize=15)
ax1.tick_params(axis='y', labelsize=15)


# target_name = ['TD 20%','TD 40%','TD 60%','TD 80%','TD 100%',]
xtd = np.array([0.2,0.4,0.6,0.8,1])
ytd = np.array([0.76, 0.784, 0.805, 0.823, 0.846]) # TD_mean
error = np.array([0.048989795, 0.04241825,  0.04556451,  0.038122004, 0.04556451]) # TD_std



ax1.plot(xtd, ytd, label='TD', c=clrs[1])
ax1.fill_between(xtd,ytd-error,ytd+error, alpha=0.3, facecolor=clrs[1])

plt.title("Test results on partial completion trajectory for TD user")
plt.savefig('../results/visualization/partial/partial_td.png')
plt.close()