import matplotlib.pyplot as plt

import pandas as pd

for seed in range(1, 20):
    for t in ["False"]:
        for g in ["True"]:
                title = f"gaze: {g}, touch: {t}, Seed: {seed}"
                # data_path = f"/home/deep/Desktop/ASDvsTDNP/ASD_SELF/Results_BOTH_First_Mar31_1505/Save_MAR31_1547_sd{seed}_t_{t}_g_{g}_lstm_True.csv"
                data_path = f"/home/deep/Desktop/ASDvsTDNP/ASD_SELF/BefApr4Res/Save_April2_sd{seed}_t_{t}_g_{g}_lstm_False.csv"
                # data_path = "/home/dp7972/Desktop/ASD_TD_WORk/ASD_SELF/SaveData_MAR28_OnlyTouch_1836_64D.csv"
                # title = "Both"

                data1 = pd.read_csv(data_path)
                columns = data1.columns
                print("columns: ", columns)
                dnp = data1.to_numpy()  # [:300]
                # print(dnp)

                # for metric in ['Train loss', 'test loss']:
                #     plt.plot(data1[f'{metric}'].to_numpy(), label=f"{metric}")
                # plt.xlabel("Epoch")
                # plt.title(title)
                # plt.legend()
                # plt.grid(color='g', linestyle='-', linewidth=.1, )
                # plt.savefig(f"old_{title}_loss.png")
                # # plt.show()


                for metric in ['Train acc', 'test acc']:
                    plt.plot(data1[f'{metric}'].to_numpy(), label=f"{metric}")
                # plt.axhline(y = 0.5, color = 'r', linestyle = '-')
                plt.xlabel("Epoch")
                plt.title(title)
                plt.legend()
                plt.savefig(f"old_{title}_acc.png")
                plt.show()

                # for metric in ['test ASD acc', 'test TD acc', 'test acc']:
                #     plt.plot(data1[f'{metric}'].to_numpy(), label=f"{metric}")
                # # plt.axhline(y = 0.5, color = 'r', linestyle = '-')
                # plt.xlabel("Epoch")
                # plt.title(title)
                # plt.legend()
                # plt.savefig(f"old_{title}_acc.png")
                # plt.show()

