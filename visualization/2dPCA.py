from sklearn.decomposition import PCA, KernelPCA
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


n_components = 7
offset = 0.5
pca = KernelPCA(n_components=n_components)
X_r = pca.fit(X).transform(X)

# lda = LinearDiscriminantAnalysis(n_components=1)
# X_r2 = lda.fit(X, y).transform(X)

# Percentage of variance explained for each components
# print(
#     "explained variance ratio (first two components): %s"
#     % str(pca.explained_variance_ratio_)
# )




colors = ["navy", "turquoise"]
lw = 2
index = 0
for pca1 in range(n_components):
    for pca2 in range(n_components):

        df_pca_ASD = []
        df_pca_TD = []

        index += 1
        if pca1 == pca2:
            continue

        # plt.figure(figsize=(5, 5))
        # ax = plt.subplot(n_components,n_components,index)
        for color, i, target_name in zip(colors, [0, 1], target_names):
            if i==0:
                # plt.scatter(
                #     X_r[y == i, pca1]+offset, X_r[y == i, pca2]+offset, color=color, alpha=0.8, lw=lw, label=target_name
                # )
                for (x1, y1) in zip(X_r[y == i, pca1] + offset, X_r[y == i, pca2] + offset):
                    df_pca_TD.append(np.array([x1,y1]))
            else:
                # plt.scatter(
                #     X_r[y == i, pca1], X_r[y == i, pca2], color=color, alpha=0.8, lw=lw, label=target_name
                # )
                for (x2, y2) in zip(X_r[y == i, pca1], X_r[y == i, pca2]):
                    df_pca_ASD.append(np.array([x2,y2]))

        # plt.legend(loc="best", shadow=False, scatterpoints=1)
        # plt.title(f"PCA of components {pca1} and {pca2}")
        # # plt.show()
        # plt.savefig(f'../results/visualization/pca_cluster/pca_result/deepsetPCA of components {pca1} and {pca2}.png')
        # plt.close()


        df_pca_TD = pd.DataFrame(np.array(df_pca_TD))
        df_pca_TD.to_csv(f'../results/visualization/pca_cluster/pca_compo_store/TD/pca_compo_{pca1}_{pca2}_TD.csv', index=None,
                     header=None)

        df_pca_ASD = pd.DataFrame(np.array(df_pca_ASD))
        df_pca_ASD.to_csv(f'../results/visualization/pca_cluster/pca_compo_store/ASD/pca_compo_{pca1}_{pca2}_ASD.csv',
                         index=None,
                         header=None)

# plt.show()
# plt.savefig(f'../results/visualization/pca_cluster/attention_deepset_pca.png')
# plt.close()
