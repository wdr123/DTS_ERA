from sklearn.decomposition import PCA, TruncatedSVD, KernelPCA
from sklearn.manifold import TSNE
import os
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt

# attention_heatmap_path = '../results/visualization/results_heatmap'
# whole_attention_data = []


# for usr_level_folder in os.listdir(attention_heatmap_path):
#     usr_level_path = os.path.join(attention_heatmap_path, usr_level_folder)
#     for attn_file in os.listdir(usr_level_path):
#         attn_file_path = os.path.join(usr_level_path, attn_file)
#         image = Image.open(attn_file_path).convert("L").resize((20,25))
#         arr = np.asarray(image).flatten().squeeze()
#         if "ASD" in usr_level_folder:
#             whole_attention_data.append(arr)
#         elif "TD" in usr_level_folder:
#             whole_attention_data.append(arr)
#         else:
#             continue
#
# whole_attention_data = np.array(whole_attention_data)
# print(whole_attention_data.shape)
# np.savetxt(f'../results/visualization/tsne_cluster/attention_fig.csv', whole_attention_data, delimiter=',')


attention_fig_path = '../results/visualization/tsne_cluster/attention_fig.csv'
attention_label_path = '../results/visualization/tsne_cluster/attention_fig_label.csv'
df_attn = pd.read_csv(attention_fig_path, header=None)
X = df_attn
y = pd.read_csv(attention_label_path,header=None)
y = y.to_numpy().squeeze()
# print(np.sum(y))
# print(X.shape,y.shape)

target_names = ["TD", "ASD"]


n_components = 3
svd = TruncatedSVD(n_components=n_components)
# tsne = TSNE(n_components=n_components,init='pca',learning_rate='auto')
X_r = svd.fit(X).transform(X)
# X_r = tsne.fit_transform(X)
print(X_r.shape)

# lda = LinearDiscriminantAnalysis(n_components=1)
# X_r2 = lda.fit(X, y).transform(X)

# Percentage of variance explained for each components
# print(
#     "explained variance ratio (first two components): %s"
#     % str(tsne.explained_variance_ratio_)
# )



plt.figure(figsize=(24,24))
colors = ["navy", "turquoise"]
lw = 2
index = 0
for svd1 in range(n_components):
    for svd2 in range(n_components):
        index += 1
        if svd1 == svd2:
            continue


        ax = plt.subplot(n_components,n_components,index)
        for color, i, target_name in zip(colors, [0, 1], target_names):
            ax.scatter(
                X_r[y == i, svd1], X_r[y == i, svd2], color=color, alpha=0.8, lw=lw, label=target_name
            )

        ax.legend(loc="best", shadow=False, scatterpoints=1)
        ax.set_title(f"svd of components {svd1} and {svd2}")

# plt.show()
plt.savefig(f'../results/visualization/svd_cluster/attention_fig_svd.png')
plt.close()
