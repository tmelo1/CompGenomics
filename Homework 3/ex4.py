
# coding: utf-8

# In[91]:

import numpy as np
from scipy.stats import pearsonr
from sklearn.covariance import GraphLasso

#threshold 1 = 0.25
#threshold 2 = 0.3
#threshold 3 = 0.

data = np.genfromtxt('expr_ceph_utah_1000.txt', delimiter='\t', skip_header=1)[:,1:]
transposed_data = np.transpose(data)


sim_matrix = np.zeros((1000, 1000))

for i in range(0,1000):
    for j in range(0,1000):
        sim_matrix[i][j] = np.absolute(pearsonr(transposed_data[i], transposed_data[j])[0])
        
expression_probes_5 = sim_matrix[:5,:5]
rng = np.arange(0.25, 0.85, 0.05)
networks = {x: np.zeros((1000,1000)) for x in rng}
for key in networks:
    for i in range(0,1000):
        for j in range(0,1000):
            if sim_matrix[i][j] > key:
                networks[key][i][j] = 1
                
degree = np.zeros((10000, 12))
j = 0
for key in networks:
    for i in range(0,1000):
        degree[i][j] = np.sum(networks[key], axis=1)[i]
    j += 1

    


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



