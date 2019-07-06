import numpy as np
import matplotlib.pyplot as plt

num_of_sam = 50
std_dv = 1.8

x1 = np.random.randn(2,num_of_sam)*std_dv + 2.5
x2 = np.random.randn(2,num_of_sam)*std_dv + 7.5
X = np.hstack((x1, x2))

t1 = np.vstack((np.zeros((1,num_of_sam)),np.ones((1,num_of_sam))))
t2 = np.vstack((np.ones((1,num_of_sam)),np.zeros((1,num_of_sam))))
T = np.hstack((t1, t2))

dataset = [X,T]


# plt.plot(x1[0], x1[1], 'o')
# plt.plot(x2[0], x2[1], 'o')
# plt.show()

import pickle

dataset_dir = 'C:\\Users\FutoshiTsutsumi\Desktop\python_test\\tensorflow\dataset_sample\pickle'
save_file = dataset_dir + '\dataset_linear_class.pkl'

with open(save_file, 'wb') as f:
    pickle.dump(dataset, f) 