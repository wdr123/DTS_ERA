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


deepset_asd = '../results/visualization/result_deepset/ASD.csv'
deepset_td = '../results/visualization/result_deepset/TD.csv'
df_asd = pd.read_csv(deepset_asd,header=None) # 75*1024
df_td = pd.read_csv(deepset_td,header=None) # 75*1024
df_asd.insert(0,'label',1)  # 75*1025
df_td.insert(0,'label',0)   # 75*1025
frames = [df_asd, df_td]
result = pd.concat(frames)
df_attn = result.sample(frac=1).reset_index(drop=True)

# print(df_attn)
X = df_attn.loc[:,0:]
# y = pd.read_csv(attention_label_path, header=None)
y = df_attn.loc[:,'label']
# print(y)
# print(np.sum(y))
# print(X.shape,y.shape)

target_names = ["TD", "ASD"]


n_components = 3
# svd = TruncatedSVD(n_components=n_components)
tsne = TSNE(n_components=n_components,init='pca',learning_rate='auto')
# X_r = tsne.fit(X).transform(X)
X_r = tsne.fit_transform(X)
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
for tsne1 in range(n_components):
    for tsne2 in range(n_components):
        index += 1
        if tsne1 == tsne2:
            continue


        ax = plt.subplot(n_components,n_components,index)
        for color, i, target_name in zip(colors, [0, 1], target_names):
            ax.scatter(
                X_r[y == i, tsne1], X_r[y == i, tsne2], color=color, alpha=0.8, lw=lw, label=target_name
            )

        ax.legend(loc="best", shadow=False, scatterpoints=1)
        ax.set_title(f"tsne of components {tsne1} and {tsne2}")

# plt.show()
plt.savefig(f'../results/visualization/tsne_cluster/attention_deepset_tsne.png')
plt.close()
