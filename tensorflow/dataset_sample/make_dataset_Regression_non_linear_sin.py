import numpy as np
import matplotlib.pyplot as plt

num_of_sam = 40
x = np.random.uniform(0, 6, num_of_sam)

noise = np.random.normal(0, 0.2, num_of_sam) 
t = np.sin(x) + 5 + noise

dataset = [x,t]

# plt.plot(x, t, 'o')
# plt.show()

import pickle

dataset_dir = 'C:\\Users\FutoshiTsutsumi\Desktop\python_test\\tensorflow\dataset_sample\pickle'
save_file = dataset_dir + '\Regression_non_linear_sin.pkl'

with open(save_file, 'wb') as f:
    pickle.dump(dataset, f) 