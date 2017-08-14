
# coding: utf-8

# In[2]:

import numpy as np

sigma1 = np.array([[1.0, 0.4328], [0.4328, 1.0]])
sigma2 = np.array([[1.0, -0.3184], [-0.3184, 1.0]])

inv1 = np.linalg.inv(sigma1)
inv2 = np.linalg.inv(sigma2)


# In[3]:

inv1


# In[4]:

inv2


# In[ ]:



