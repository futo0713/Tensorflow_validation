import numpy as np
import matplotlib.pyplot as plt

import pickle

name = 'dataset_linear'

dataset_dir = 'C:\\Users\FutoshiTsutsumi\Desktop\python_test\\tensorflow\dataset_sample\pickle'
save_file = dataset_dir + '\dataset_linear.pkl'

with open(save_file, 'rb') as f:
    dataset = pickle.load(f)

print(dataset)

plt.plot(dataset[0],dataset[1],'o')
plt.show()