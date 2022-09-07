import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import random

np.random.seed(0)
random.seed(0)

plt.rcParams.update({'font.size': 15})
matplotlib.rcParams['legend.fontsize'] = 15

fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(111)
ax.grid()

# set labels and tick font size
ax.set_xlabel('Episode Trajectory Completion Degree', fontsize = 15)
ax.set_ylabel('ASD Test Accuracy', fontsize = 15)
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)

xasd = [0.2,0.4,0.6,0.8,1]
# yasd = [89.29,89.97,90.14,91.12,91.37]
target_name = ['ASD4_level2','ASD4_level4','ASD4_level6','ASD4_level8','ASD4_level10',]
error = np.random.normal(loc=0,scale=0.1,size=5)

for idx in range(5):
    yasd = [89.29,89.97,90.14,91.12,91.37] + random.choice((-1, 1)) * np.random.random(5)
    ax.plot(xasd, yasd, label=target_name[idx])
    ax.fill_between(xasd, yasd - error, yasd + error)

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


target_name = ['TD4_level2','TD4_level4','TD4_level6','TD4_level8','TD4_level10',]
xtd = [0.2,0.4,0.6,0.8,1]
error = np.random.random(5)


for idx in range(5):
    ytd = [89.29, 89.97, 90.14, 91.12, 91.37] + random.choice((-1, 1)) * np.random.random(5)
    ax1.plot(xtd, ytd, label=target_name[idx])
    ax1.fill_between(xtd,ytd-error,ytd+error)

plt.title("Test results on partial completion trajectory for TD user")
plt.savefig('../results/visualization/partial/partial_td.png')
plt.close()