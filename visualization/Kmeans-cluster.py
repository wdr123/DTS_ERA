import os

from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# X = np.array([[1, 2], [1, 4], [1, 0],[10, 2], [10, 4], [10, 0]])
# X_ASD = pd.read_csv('../data/cluster/cluster_ASD_FlattenedPCA.csv',header=None).to_numpy()
# X_TD = pd.read_csv('../data/cluster/cluster_TD_FlattenedPCA.csv',header=None).to_numpy()
# X_ASD_mean = pd.read_csv('../data/cluster/cluster_ASD_mean_FlattenedPCA.csv',header=None).to_numpy()
# X_TD_mean = pd.read_csv('../data/cluster/cluster_TD_mean_FlattenedPCA.csv',header=None).to_numpy()
# X_ASD_attn = pd.read_csv('../data/cluster/cluster_ASD_attn_FlattenedPCA.csv',header=None).to_numpy()
# X_TD_attn = pd.read_csv('../data/cluster/cluster_TD_attn_FlattenedPCA.csv',header=None).to_numpy()

# fig = plt.figure(figsize=(12,10))
# ax_ground = plt.subplot(1,3,1)
# ax_attn_ground = plt.subplot(1,3,2)
# ax_mean_ground = plt.subplot(1,3,3)
#
# ax_ground.scatter(X_ASD[:,0], X_ASD[:,1], alpha=0.5, label='ASD')
# ax_ground.scatter(X_TD[:,0], X_TD[:,1], alpha=0.5, label='TD')
# ax_attn_ground.scatter(X_ASD_attn[:,0], X_ASD_attn[:,1], alpha=0.5, label='ASD_attn')
# ax_attn_ground.scatter(X_TD_attn[:,0], X_TD_attn[:,1], alpha=0.5, label='TD_attn')
# ax_mean_ground.scatter(X_ASD_mean[:,0], X_ASD_mean[:,1], alpha=0.5, label='ASD_mean')
# ax_mean_ground.scatter(X_TD_mean[:,0], X_TD_mean[:,1], alpha=0.5, label='TD_mean')
# ax_ground.legend(loc="upper left")
# ax_mean_ground.legend(loc="upper left")
# ax_attn_ground.legend(loc="upper left")
# plt.savefig(f'../results/visualization/cluster_before.png')
# plt.close()
asd_dir_attn = '../results/visualization/pca_cluster/pca_compo_store/ASD'
td_dir_attn = '../results/visualization/pca_cluster/pca_compo_store/TD'

for asd_attn_pca_file, td_attn_pca_file in zip(os.listdir(asd_dir_attn),os.listdir(td_dir_attn)):
    asd_attn_path = os.path.join(asd_dir_attn, asd_attn_pca_file)
    td_attn_path = os.path.join(td_dir_attn, td_attn_pca_file)
    print(asd_attn_pca_file)
    print(td_attn_pca_file)
    # X_ASD_attn = pd.read_csv('../results/visualization/pca_cluster/pca_compo_store/pca_component_store_3_4_ASD').to_numpy()
    # X_TD_attn = pd.read_csv('../results/visualization/pca_cluster/pca_compo_store/pca_component_store_3_4_TD').to_numpy()
    # print(asd_attn_path)
    X_ASD_attn = np.genfromtxt(asd_attn_path, delimiter=',')
    X_TD_attn = np.genfromtxt(td_attn_path, delimiter=',')

    num_cluster = 8

    X_attn = np.concatenate([X_ASD_attn,X_TD_attn], axis=0)

    kmeans_attn = KMeans(n_clusters=num_cluster, random_state=0).fit(X_attn)


    # kmeans.predict([[0, 0], [12, 3]]) # (input_dim,1)
    # kmeans.cluster_centers_  # (16,2)
    # kmeans.labels_ # (6000,1)
    # print(kmeans.labels_)
    # print(kmeans.cluster_centers_)

    fig = plt.figure(figsize=(12, 10))
    symbol_map = ['^','v','<','>','1','2','+','x','X','s','p','P','*','h','H','3','4']

    for cluster_idx in range(num_cluster):
        bool_idx_attn = (kmeans_attn.labels_ == cluster_idx)
        plt.scatter(X_attn[bool_idx_attn,0], X_attn[bool_idx_attn,1], alpha=0.8, label=f'{cluster_idx}_attn')
        plt.scatter(kmeans_attn.cluster_centers_[cluster_idx,0], kmeans_attn.cluster_centers_[cluster_idx,1], alpha=0.5, marker=symbol_map[cluster_idx], s=100, label=f'{cluster_idx}_center')


    plt.legend(loc="upper left")
    path_handle = '_'.join(asd_attn_pca_file.split('_')[:-1])
    plt.savefig(f"../results/visualization/pca_cluster/kmeans_result/kmeans_{path_handle}.png")
    plt.close()