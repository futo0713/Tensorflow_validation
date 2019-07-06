import numpy as np
import matplotlib.pyplot as plt

num_of_sam = 20
x = np.random.uniform(0, 2, num_of_sam)

noise = np.random.normal(0, 0.2, num_of_sam) 
t = 3*x + 5 + noise

dataset = np.empty((0,num_of_sam))
dataset = np.vstack((dataset,np.reshape(x,(1,20))))
dataset = np.vstack((dataset,np.reshape(t,(1,20))))
print(dataset)

# plt.plot(x, t, 'o')
# plt.show()

import pickle

dataset_dir = 'C:\\Users\FutoshiTsutsumi\Desktop\python_test\\tensorflow\dataset_sample\pickle'
save_file = dataset_dir + '\dataset_linear.pkl'

with open(save_file, 'wb') as f:
    pickle.dump(dataset, f) 