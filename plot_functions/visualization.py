import sys

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

path = 'csv_file/save_mean/Save_mean_Batch16_sd1.csv'
data = pd.read_csv(path, header=0)

# print(data['Train acc'])



plt.xlabel('Epoch')
plt.ylabel('Training Acc')
plt.plot('Epoch', 'Train acc', data=data)
plt.savefig('results/mean/average1/Train acc.png')
plt.close()

plt.xlabel('Epoch')
plt.ylabel('Training Action Loss')
plt.plot('Epoch', 'Train Action Loss', data=data)
plt.savefig('results/mean/average1/Training Action Loss.png')
plt.close()

plt.xlabel('Epoch')
plt.ylabel('Training Location Loss')
plt.plot('Epoch', 'Train Location Loss', data=data)
plt.savefig('results/mean/average1/Train Location Loss.png')
plt.close()

plt.xlabel('Epoch')
plt.ylabel('Train Baseline Loss')
plt.plot('Epoch', 'Train Baseline Loss', data=data)
plt.savefig('results/mean/average1/Train Baseline Loss.png')
plt.close()


plt.xlabel('Epoch')
plt.ylabel('Test Accuracy (%)')
plt.plot('Epoch', 'Test Accuracy', data=data)
plt.savefig('results/mean/average1/Test Accuracy.png')
plt.close()




plt.xlabel('Epoch')
plt.ylabel('Test Accuracy (%)')
plt.plot('Epoch', 'Test ASD Accuracy', data=data)
plt.savefig('results/mean/average1/Test ASD Accuracy.png')
plt.close()

plt.xlabel('Epoch')
plt.ylabel('Test Accuracy (%)')
plt.plot('Epoch', 'Test TD Accuracy', data=data)
plt.savefig('results/mean/average1/Test TD Accuracy.png')
plt.close()

