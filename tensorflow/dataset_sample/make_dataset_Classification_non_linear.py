import numpy as np
import matplotlib.pyplot as plt

num_of_sam = 40
std_dv = 0.6
radius = 4

X_center = np.random.randn(num_of_sam,2)*std_dv

s = np.random.uniform(0,2*np.pi,num_of_sam)
noise = np.random.uniform(0.9, 1.1, num_of_sam)
x1 = np.sin(s)*radius*noise
x2 = np.cos(s)*radius*noise
X_circle = np.c_[x1,x2]

X = np.vstack((X_center,X_circle))


t_group1 = np.tile([0,1],(num_of_sam,1))
t_group2 = np.tile([1,0],(num_of_sam,1))
T = np.vstack((t_group1, t_group2))


dataset = [X_center,X_circle,t_group1,t_group2,X,T]


# plt.plot(X_center[:,0],X_center[:,1], 'o',color='red')
# plt.plot(X_circle[:,0],X_circle[:,1], 'o',color='blue')
# plt.show()

import pickle

dataset_dir = 'C:\\Users\FutoshiTsutsumi\Desktop\python_test\\tensorflow\dataset_sample\pickle'
save_file = dataset_dir + '\Classification_non_linear.pkl'

with open(save_file, 'wb') as f:
    pickle.dump(dataset, f) 