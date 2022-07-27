import matplotlib.pyplot as plt
from average_seed import average_seed
import os

import pandas as pd
avg_acc = 0
count = 0
base_dir = '/home/dw7445/Projects/Recurrent-Model-of-Visual-Attention/results'
plot_dir = os.path.join(base_dir, 'plots')
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

average_file = average_seed('/home/dw7445/Projects/Recurrent-Model-of-Visual-Attention/results/partial/12')
for seed in range(1,21):
    for t in ["True", "False"][:1]:
        for g in [ "True","False",][:1]:
            for lstm in ["True", "False"][:1]:
                title = f"gaze: {g}, touch: {t}, lstm: {lstm}, Seed: {seed}"
                print(title)
                # data_path = f"/home/deep/Desktop/ASDvsTDNP/ASD_SELF/Incomplete_Apr1/Save_April1_MIL0732_sd{seed}_t_{t}_g_{g}_lstm_{lstm}.csv"
                # data_path = f"/home/deep/Desktop/ASDvsTDNP/ASD_SELF/ResApr12NoPD/Save_Comp_Apr6_ats_noPD_Batch8_sd{seed}_t_{t}_g_{g}_lstm_{lstm}.csv"
                data_path = f"/home/dw7445/Projects/Recurrent-Model-of-Visual-Attention/results/partial/12/Save_partial_Parital12_Batch16_sd{seed}.csv"
                # data_path = f"/home/deep/Desktop/ASDvsTDNP/ASD_SELF/ResApr22/Save_Mil.Apr19.NoPD._Batch8_sd{seed}_t_{t}_g_{g}_lstm_{lstm}.csv"
                #data_path = f"/home/deep/Desktop/ASDvsTDNP/ASD_SELF/ResApr22/Save_Mil.Apr19.1734.TK.NoPD._TK_sd{seed}_t_{t}_g_{g}_lstm_{lstm}"
                #data_path += "/results.csv"
                
                # title = "Both"
                if seed in range(1,20):
                    data1 = pd.read_csv(data_path)
                else:
                    data1 = average_file
                    title = f"gaze: {g}, touch: {t}, lstm: {lstm}, Seed: average"
                columns = data1.columns
                # print("columns: ", columns)
                dnp = data1.to_numpy()#[:300]
                # print(dnp)


                # for metric in ['Train loss', "test loss"]:
                #     plt.plot([float(x) for x in data1[f'{metric}'].to_numpy()[:1000]], label=f"{metric}")
                # plt.xlabel("Epoch")
                # plt.title(title)
                # plt.legend()
                # plt.grid(color='g', linestyle='-', linewidth=.1, )
                # plt.savefig(f"new_mil_{title}_loss.png")
                #
                # plt.show()
                
                # for metric in ["test ASD acc", "test TD acc"]:
                #     plt.plot(data1[f'{metric}'].to_numpy(), label=f"{metric}")
                # # plt.axhline(y = 0.5, color = 'r', linestyle = '-')
                # plt.xlabel("Epoch")
                # plt.title(title)
                # plt.legend()
                # plt.savefig(f"new_mil_{title}_acc_fine.png")
                #
                # plt.show()

                num_iteration_av = 1
                for metric in ['Train acc', 'Test Accuracy (%)']:
                    to_plot = [float(x) for x in data1[f'{metric}'].to_numpy()[:1000]]
                    if metric  == 'Test Accuracy (%)':
                        avg_acc += to_plot[998]
                        count += 1
                        # print(to_plot)
                    plot_this = []
                    ct_10, sm  = 0, 0
                    for x in to_plot:
                        ct_10 += 1
                        sm += x
                        if ct_10 == num_iteration_av:
                            plot_this.append(sm/num_iteration_av)
                            # print(sm/num_iteration_av)
                            ct_10, sm = 0, 0
                    plt.plot(plot_this, label=f"{metric}")
                # plt.axhline(y = 0.5, color = 'r', linestyle = '-')
                plt.xlabel(f"Epoch (*{num_iteration_av})")
                plt.title(title)
                plt.legend()
                plt.savefig(os.path.join(plot_dir, f"Jul25_{title}_acc.png"))
                plt.clf()

                for metric in ['Train Action Loss', 'Test Action Loss']:
                    to_plot = [float(x) for x in data1[f'{metric}'].to_numpy()[:1000]]
                    plot_this = []
                    ct_10, sm = 0, 0
                    for x in to_plot:
                        ct_10 += 1
                        sm += x
                        if ct_10 == num_iteration_av:
                            plot_this.append(sm / num_iteration_av)
                            ct_10, sm = 0, 0
                    plt.plot(plot_this, label=f"{metric}")
                plt.xlabel(f"Epoch*{num_iteration_av})")
                plt.title(title)
                plt.legend()
                plt.savefig(os.path.join(plot_dir, f"Jul25_{title}_action_loss.png"))
                plt.clf()

                for metric in ['Train Location Loss', 'Test Location Loss']:
                    to_plot = [float(x) for x in data1[f'{metric}'].to_numpy()[:1000]]
                    plot_this = []
                    ct_10, sm = 0, 0
                    for x in to_plot:
                        ct_10 += 1
                        sm += x
                        if ct_10 == num_iteration_av:
                            plot_this.append(sm / num_iteration_av)
                            ct_10, sm = 0, 0
                    plt.plot(plot_this, label=f"{metric}")
                plt.xlabel(f"Epoch*{num_iteration_av})")
                plt.title(title)
                plt.legend()
                plt.savefig(os.path.join(plot_dir, f"Jul25_{title}_location_loss.png"))
                plt.clf()



print("AVG : ", avg_acc/count)