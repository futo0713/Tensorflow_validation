import numpy as np
import matplotlib.pyplot as plt

num_of_sam = 40
std_dv = 1.8

group1 = np.array([2.5,2.5])+np.random.randn(num_of_sam, 2)*std_dv
group2 = np.array([7.5,7.5])+np.random.randn(num_of_sam, 2)*std_dv
X = np.vstack((group1, group2))

t_group1 = np.tile([0,1],(num_of_sam,1))
t_group2 = np.tile([1,0],(num_of_sam,1))
T = np.vstack((t_group1, t_group2))

dataset = [group1,group2,t_group1,t_group2,X,T]


# plt.plot(group1[:,0], group1[:,1], 'o')
# plt.plot(group2[:,0], group2[:,1], 'o')
# plt.show()

import pickle

dataset_dir = 'C:\\Users\FutoshiTsutsumi\Desktop\python_test\\tensorflow\dataset_sample\pickle'
save_file = dataset_dir + '\Classification_linear.pkl'

with open(save_file, 'wb') as f:
    pickle.dump(dataset, f) 