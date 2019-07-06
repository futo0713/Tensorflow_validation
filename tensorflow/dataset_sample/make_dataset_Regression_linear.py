import numpy as np
import matplotlib.pyplot as plt

num_of_sam = 30
x = np.random.uniform(0, 4, num_of_sam)

noise = np.random.normal(0, 0.5, num_of_sam) 
t = 2*x+5+noise

dataset = [x,t]

# plt.plot(x, t, 'o')
# plt.show()

import pickle

dataset_dir = 'C:\\Users\FutoshiTsutsumi\Desktop\python_test\\tensorflow\dataset_sample\pickle'
save_file = dataset_dir + '\Regression_linear.pkl'

with open(save_file, 'wb') as f:
    pickle.dump(dataset, f) 