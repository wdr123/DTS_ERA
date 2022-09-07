from sklearn.decomposition import PCA
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
# np.savetxt(f'../results/visualization/pca_cluster/attention_fig.csv', whole_attention_data, delimiter=',')


attention_fig_path = '../results/visualization/pca_cluster/attention_fig.csv'
attention_label_path = '../results/visualization/pca_cluster/attention_fig_label.csv'
df_attn = pd.read_csv(attention_fig_path,header=None)
X = df_attn
y = pd.read_csv(attention_label_path, header=None)
y = y.to_numpy().squeeze()
print(np.sum(y))
print(X.shape,y.shape)

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
plt.savefig(f'../results/visualization/pca_cluster/attention_fig_pca.png')
plt.close()
