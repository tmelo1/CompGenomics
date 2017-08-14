
# coding: utf-8

# In[168]:

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sets import Set

#Muscle-skeletal - 0
#Lung - 1
#Thyroid - 2
#Adipose-subcutaneous - 3
#Whole blood - 4


sample_labels = np.genfromtxt('class_labels.txt', delimiter='\t', skip_header=1, usecols=1)
data = np.genfromtxt('expr.txt', delimiter='\t', skip_header=1, unpack=True)[1:,:]
init_clusters = np.array(data[:5])
kmeans_5 = KMeans(n_clusters=5, max_iter=10, n_init=1, init=init_clusters).fit(data)
cluster_sizes = np.bincount(kmeans.labels_)
cluster_centers_5means = kmeans_5.cluster_centers_
bic_values = []

for k in range(2, 11):
    k_init_clusters = np.array(data[:k])
    kmeans = KMeans(n_clusters=k, max_iter=10, n_init=1, init=k_init_clusters).fit(data)
    bic_k = 2*(kmeans.inertia_) + k*100*np.log(1816)
    bic_values.append(bic_k)
    
plt.plot(range(2,11), bic_values, "o")
plt.ylabel("BIC")
plt.xlabel("k")
plt.show()


kmeans_6 = KMeans(n_clusters=6, max_iter=10, n_init=1, init=np.array(data[:6])).fit(data)
pca_2 = PCA(n_components=2)
pca_2.fit(data)
pca_2_transform = pca_2.transform(data)

plt.figure()
plt.plot(pca_2_transform[:,:1], pca_2_transform[:,1:], "+")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.show()



# In[138]:

kmeans_6.labels_


# In[91]:

data.shape


# In[124]:

bic_values


# In[147]:




# In[ ]:




# In[ ]:



