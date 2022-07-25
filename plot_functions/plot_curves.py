import matplotlib.pyplot as plt

import pandas as pd

# data_path = "/home/dp7972/Desktop/ASD_TD_WORk/ASD_SELF/SaveData_MAR28_OnlyGAZE_TRUE_1851_64D.csv"
#data_path = "/home/dp7972/Desktop/ASD_TD_WORk/ASD_SELF/Save_Apr2_1758_LSTWITHOUT pupil diameter_sd12_t_True_g_True_lstm_True.csv"
#data_path = "/home/dp7972/Desktop/ASD_TD_WORk/ASD_SELF/Save_NoTouch_DPDR1S12_Batch8_sd12_t_False_g_True_lstm_False.csv"
data_path = "/home/dp7972/Desktop/ASD_TD_WORk/ASD_SELF/Save_LSTMNoTcPDR1S12_Batch8_sd12_t_False_g_True_lstm_True.csv"
# data_path = "/home/dp7972/Desktop/ASD_TD_WORk/ASD_SELF/SaveData_MAR28_OnlyTouch_1836_64D.csv"
data_path = "/home/deep/Desktop/ASDvsTDNP/ASD_SELF/Save_Gaze_only_again_apr14_Batch8_sd18_t_False_g_True_lstm_False"
data_path = "/home/deep/Desktop/ASDvsTDNP/ASD_SELF/Save_Gaze_only_again_apr15_4dGazeLSTM_Batch8_sd18_t_False_g_True_lstm_True"
data_path = "/home/deep/Desktop/ASDvsTDNP/ASD_SELF/Save_Gaze_only_again_apr15_only3dGazeOther2_Batch8_sd18_t_False_g_True_lstm_True"
data_path += "/results.csv"
title = "Right Eye Gaze"

data1 = pd.read_csv(data_path)
columns = data1.columns
print("columns: ", columns)
dnp = data1.to_numpy()#[:300]
print(dnp)

for metric in ['Train acc', 'test acc']:
    to_plot = [float(x) for x in data1[f'{metric}'].to_numpy()[:1000]]
    plot_this = []
    ct_10, sm = 0, 0
    for x in to_plot:
        ct_10 += 1
        sm += x
        if ct_10 == 10:
            plot_this.append(sm / 10)
            ct_10, sm = 0, 0
    plt.plot(plot_this, label=f"{metric}")
plt.xlabel("Epoch")
plt.title(title)
plt.legend()
plt.show()

for metric in ['Train loss', 'test loss']:
    to_plot = [float(x) for x in data1[f'{metric}'].to_numpy()[:1000]]
    plot_this = []
    ct_10, sm = 0, 0
    for x in to_plot:
        ct_10 += 1
        sm += x
        if ct_10 == 10:
            plot_this.append(sm / 10)
            ct_10, sm = 0, 0
    plt.plot(plot_this, label=f"{metric}")
plt.xlabel("Epoch")
plt.title(title)
plt.legend()
plt.show()
