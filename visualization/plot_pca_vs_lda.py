"""
=======================================================
Comparison of LDA and PCA 2D projection of Iris dataset
=======================================================

The Iris dataset represents 3 kind of Iris flowers (Setosa, Versicolour
and Virginica) with 4 attributes: sepal length, sepal width, petal length
and petal width.

Principal Component Analysis (PCA) applied to this data identifies the
combination of attributes (principal components, or directions in the
feature space) that account for the most variance in the data. Here we
plot the different samples on the 2 first principal components.

Linear Discriminant Analysis (LDA) tries to identify attributes that
account for the most variance *between classes*. In particular,
LDA, in contrast to PCA, is a supervised method, using known class labels.

"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# iris = datasets.load_iris()
# attention_data_path = '../results/visualization/result_attention_data'
# whole_attention_data = []
#
#
# for usr_level_folder in os.listdir(attention_data_path):
#     usr_level_path = os.path.join(attention_data_path, usr_level_folder)
#     for attn_file in os.listdir(usr_level_path):
#         attn_file_path = os.path.join(usr_level_path, attn_file)
#         df_attn = pd.read_csv(attn_file_path, header=None)
#         df_attn = df_attn.T
#         df_attn.columns = ["left_gaze_x", "left_gaze_y", "right_gaze_x", "right_gaze_y", "touch_x", "touch_y", "touch_hard", "duration"]
#         if "ASD" in usr_level_folder:
#             label = [1 for _ in range(len(df_attn))]
#         else:
#             label = [0 for _ in range(len(df_attn))]
#         df_attn['label'] = label
#         array_attn = df_attn.to_numpy()
#         # print(array_attn.shape)
#         whole_attention_data.append(array_attn)
#
# whole_attention_data = np.concatenate(whole_attention_data, axis=0)
# print(whole_attention_data.shape)
# df_attn = pd.DataFrame(whole_attention_data, columns=["left_gaze_x", "left_gaze_y", "right_gaze_x",\
#                                                       "right_gaze_y", "touch_x", "touch_y", "touch_hard", "duration", "label"])
#
# df_attn.to_csv('../results/visualization/pca_cluster/attention_all_w_label.csv', index=None)

df_attn = pd.read_csv('../results/visualization/pca_cluster/attention_all_w_label.csv')
X = df_attn.loc[:, ["left_gaze_x", "left_gaze_y", "right_gaze_x", "right_gaze_y", "touch_x", "touch_y", "touch_hard", "duration"]]
y = df_attn.loc[:, "label"]
target_names = ["TD", "ASD"]


n_components = 7
pca = PCA(n_components=n_components)
X_r = pca.fit(X).transform(X)

# lda = LinearDiscriminantAnalysis(n_components=1)
# X_r2 = lda.fit(X, y).transform(X)

# Percentage of variance explained for each components
print(
    "explained variance ratio (first two components): %s"
    % str(pca.explained_variance_ratio_)
)



plt.figure(figsize=(24,24))
colors = ["navy", "turquoise"]
lw = 2
index = 0
for pca1 in range(n_components):
    for pca2 in range(n_components):
        index += 1
        if pca1 == pca2:
            continue


        ax = plt.subplot(n_components,n_components,index)
        for color, i, target_name in zip(colors, [0, 1], target_names):
            ax.scatter(
                X_r[y == i, pca1], X_r[y == i, pca2], color=color, alpha=0.8, lw=lw, label=target_name
            )

        ax.legend(loc="best", shadow=False, scatterpoints=1)
        ax.set_title(f"PCA of components {pca1} and {pca2}")

# plt.show()
plt.savefig(f'../results/visualization/pca_cluster/attention_data_pca.png')
plt.close()


# pca1_show = [3,3,5,5]
# pca2_show = [4,5,2,3]
#
# for i in range(2):
#     for show1, show2 in zip(pca1_show, pca2_show):
#         X_copy = np.concatenate([X_r[y == i, show1][:, np.newaxis],X_r[y == i, show2][:, np.newaxis]], axis=1)
#         if i == 0:
#             np.savetxt(f'../results/visualization/pca_cluster/pca_component_store_{show1}_{show2}_TD', X_copy, delimiter=',')
#         else:
#             np.savetxt(f'../results/visualization/pca_cluster/pca_component_store_{show1}_{show2}_ASD', X_copy, delimiter=',')


# colors = ["navy", "turquoise"]
# lw = 2
# index = 0
#
# for pca1, pca2 in zip(pca1_show, pca2_show):
#     index += 1
#     plt.figure(figsize=(12, 10))
#
#     for color, i, target_name in zip(colors, [0, 1], target_names):
#         plt.scatter(
#             X_r[y == i, pca1], X_r[y == i, pca2], color=color, alpha=0.8, lw=lw, label=target_name
#         )
#
#     plt.legend(loc="best", shadow=False, scatterpoints=1)
#     plt.title(f"PCA of components {pca1} and {pca2}")
#     plt.savefig(f'../results/visualization/pca_cluster/kmeans_result/attention_pca_{pca1}_{pca2}.png')
#     plt.close()

