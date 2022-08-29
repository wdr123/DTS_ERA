from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os

# X = np.array([[1, 2], [1, 4], [1, 0],[10, 2], [10, 4], [10, 0]])
dir_attn = '../data/cluster/attn_new'
dir_origin = '../data/cluster/origin_new'

X_ASD= []
X_TD = []
X_ASD_attn = []
X_TD_attn = []

for file in os.listdir(dir_attn):
    data_path = os.path.join(dir_attn,file)
    one_data = pd.read_csv(data_path,header=None).to_numpy()
    if 'ASD' in file:
        X_ASD_attn.append(one_data)
    elif 'TD' in file:
        X_TD_attn.append(one_data)

for file in os.listdir(dir_origin):
    data_path = os.path.join(dir_origin, file)
    one_data = pd.read_csv(data_path,header=None).to_numpy()
    if 'ASD' in file:
        X_ASD.append(one_data)
    elif 'TD' in file:
        X_TD.append(one_data)

X_TD = np.concatenate(X_TD, axis=0)
X_ASD = np.concatenate(X_ASD, axis=0)
X_TD_attn = np.concatenate(X_TD_attn, axis=0)
X_ASD_attn = np.concatenate(X_ASD_attn, axis=0)

fig = plt.figure(figsize=(12,10))
ax_ground = plt.subplot(1,2,1)
ax_attn_ground = plt.subplot(1,2,2)
# ax_mean_ground = plt.subplot(1,3,3)

ax_ground.scatter(X_ASD[:,0], X_ASD[:,1], alpha=0.5, label='ASD')
ax_ground.scatter(X_TD[:,0], X_TD[:,1], alpha=0.5, label='TD')
ax_attn_ground.scatter(X_ASD_attn[:,0], X_ASD_attn[:,1], alpha=0.5, label='ASD_attn')
ax_attn_ground.scatter(X_TD_attn[:,0], X_TD_attn[:,1], alpha=0.5, label='TD_attn')
# ax_mean_ground.scatter(X_ASD_mean[:,0], X_ASD_mean[:,1], alpha=0.5, label='ASD_mean')
# ax_mean_ground.scatter(X_TD_mean[:,0], X_TD_mean[:,1], alpha=0.5, label='TD_mean')
ax_ground.legend(loc="upper left")
# ax_mean_ground.legend(loc="upper left")
ax_attn_ground.legend(loc="upper left")
plt.savefig(f'../results/visualization/cluster_before.png')
plt.close()


num_cluster = 2
X = np.concatenate([X_ASD,X_TD], axis=0)
# X_mean = np.concatenate([X_ASD_mean,X_TD_mean], axis=0)
X_attn = np.concatenate([X_ASD_attn,X_TD_attn], axis=0)
kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(X)
# kmeans_mean = KMeans(n_clusters=num_cluster, random_state=0).fit(X_mean)
kmeans_attn = KMeans(n_clusters=num_cluster, random_state=0).fit(X_attn)

# kmeans.predict([[0, 0], [12, 3]]) # (input_dim,1)
# kmeans.cluster_centers_  # (16,2)
# kmeans.labels_ # (6000,1)
# print(kmeans.labels_)
# print(kmeans.cluster_centers_)

fig = plt.figure(figsize=(24,20))
ax = plt.subplot(1,2,1)
ax_attn = plt.subplot(1,2,2)
# ax_mean = plt.subplot(1,3,3)
symbol_map = ['^','v','<','>','1','2','+','x','X','s','p','P','*','h','H','3','4']

for cluster_idx in range(num_cluster):
    bool_idx = (kmeans.labels_ == cluster_idx)
    # bool_idx_mean = (kmeans_mean.labels_ == cluster_idx)
    bool_idx_attn = (kmeans_attn.labels_ == cluster_idx)
    ax.scatter(X[bool_idx,0], X[bool_idx,1], alpha=0.5, label=f'{cluster_idx}')
    ax.scatter(kmeans.cluster_centers_[cluster_idx,0], kmeans.cluster_centers_[cluster_idx,1], alpha=0.5, marker=symbol_map[cluster_idx], s=100, label=f'{cluster_idx}_center')
    ax_attn.scatter(X_attn[bool_idx_attn,0], X_attn[bool_idx_attn,1], alpha=0.5, label=f'{cluster_idx}_attn')
    ax_attn.scatter(kmeans_attn.cluster_centers_[cluster_idx,0], kmeans_attn.cluster_centers_[cluster_idx,1], alpha=0.5, marker=symbol_map[cluster_idx], s=100, label=f'{cluster_idx}_center')
    # ax_mean.scatter(X_mean[bool_idx_mean,0], X_mean[bool_idx_mean,1], alpha=0.5, label=f'{cluster_idx}_mean')
    # ax_mean.scatter(kmeans_mean.cluster_centers_[cluster_idx,0], kmeans_mean.cluster_centers_[cluster_idx,1], alpha=0.5, marker=symbol_map[cluster_idx], s=100, label=f'{cluster_idx}_center')


ax.legend(loc="upper left")
# ax_mean.legend(loc="upper left")
ax_attn.legend(loc="upper left")
plt.savefig(f'../results/visualization/cluster_after.png')
plt.close()