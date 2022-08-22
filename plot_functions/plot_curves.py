import matplotlib.pyplot as plt
import matplotlib

import pandas as pd



data_path_attn = "/home/dw7445/Projects/Recurrent-Model-of-Visual-Attention/results/partial/12/Save_partial_latent256_combine_sequential_selen10_msize10_time_step5_sd0.csv"
data_path_woattn = "/home/dw7445/Projects/Recurrent-Model-of-Visual-Attention/results/partial/12/Save_partial_latent256_no_attention_combine_selen10_msize10_time_step5_sd0.csv"
data_path_128 = "/home/dw7445/Projects/Recurrent-Model-of-Visual-Attention/results/partial/12/Save_partial_latent128_combine_sequential_selen10_msize10_time_step5_sd0.csv"
data_path_256 = "/home/dw7445/Projects/Recurrent-Model-of-Visual-Attention/results/partial/12/Save_partial_latent256_combine_sequential_selen10_msize10_time_step5_sd0.csv"
data_path_512 = "/home/dw7445/Projects/Recurrent-Model-of-Visual-Attention/results/partial/12/Save_partial_latent512_combine_sequential_selen10_msize10_time_step5_sd0.csv"

data_path_30 = "/home/dw7445/Projects/Recurrent-Model-of-Visual-Attention/results/partial/12/Save_partial_latent128_combine_combine_selen10_msize2_time_step5_sd2.csv"
data_path_40 = "/home/dw7445/Projects/Recurrent-Model-of-Visual-Attention/results/partial/12/Save_partial_latent256_combine_combine_selen10_msize2_time_step5_sd2.csv"
data_path_50 = "/home/dw7445/Projects/Recurrent-Model-of-Visual-Attention/results/partial/12/Save_partial_latent512_combine_sequential_selen10_msize10_time_step5_sd2.csv"
data_path_60 = "/home/dw7445/Projects/Recurrent-Model-of-Visual-Attention/results/partial/12/Save_partial_latent256_combine_sequential_selen10_msize10_time_step5_sd0.csv"
data_path_70 = "/home/dw7445/Projects/Recurrent-Model-of-Visual-Attention/results/partial/12/Save_partial_latent512_combine_sequential_selen10_msize10_time_step5_sd0.csv"

latent_dim = [128,256,512]
input_traj = [30,40,50,60,70]
title = "Test Classification Accuracy for latent dimension (128,256,512)"
# title = "Test Classification Accuracy w/ and w/o reinforced attention mechanism"
# title = "Test Classification Accuracy for different input trajectory length (30,40,50,60,70)"
# font = {'family' : 'normal',
#         'weight' : 'bold',
#         'size'   : 12}
#
# matplotlib.rc('font', **font)
plt.rcParams.update({'font.size': 15})
matplotlib.rcParams['legend.fontsize'] = 15

# data1 = pd.read_csv(data_path)
# columns = data1.columns
#     print("columns: ", columns)
# dnp = data1.to_numpy()#[:300]
# print(dnp)
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
ax.grid()

# set labels and font size
ax.set_xlabel("Epoch", fontsize = 15)
ax.set_ylabel("Test Accuracy", fontsize = 15)
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)


for idx, data_path in enumerate([data_path_128, data_path_256, data_path_512]):
# for idx, data_path in enumerate([data_path_attn, data_path_woattn]):
# for idx, data_path in enumerate([data_path_30, data_path_40, data_path_50, data_path_60, data_path_70]):
    data1 = pd.read_csv(data_path)

    for metric in ['Test Accuracy']:
        # if idx == 1:
        #     metric = 'Test TD Accuracy'
        to_plot = [float(x) for x in data1[f'{metric}'].to_numpy()[:2000]]
        plot_this = []
        ct_10, sm = 0, 0
        for x in to_plot:
            ct_10 += 1
            sm += x
            if ct_10 == 10:
                plot_this.append(sm / 10)
                ct_10, sm = 0, 0

        # plt.plot(plot_this, label=f"input_trajectory_length_{input_traj[idx]}")
        plt.plot(plot_this, label=f"latent_dimension_{latent_dim[idx]}")
        # if idx == 0:
        #     ax.plot(plot_this, label=f"w/ attention")
        # else:
        #     ax.plot(plot_this, label=f"w/o attention")
# plt.xlabel("Epoch")
# plt.ylabel("Test Accuracy")
plt.title(title)
plt.legend()

plt.savefig('../results/visualization/latent_dim.png')
# plt.savefig('../results/visualization/attention.png')
# plt.savefig('../results/visualization/inp_traj.png')







# for metric in ['Train loss', 'test loss']:
#     to_plot = [float(x) for x in data1[f'{metric}'].to_numpy()[:1000]]
#     plot_this = []
#     ct_10, sm = 0, 0
#     for x in to_plot:
#         ct_10 += 1
#         sm += x
#         if ct_10 == 10:
#             plot_this.append(sm / 10)
#             ct_10, sm = 0, 0
#     plt.plot(plot_this, label=f"{metric}")
# plt.xlabel("Epoch")
# plt.title(title)
# plt.legend()
# plt.show()
