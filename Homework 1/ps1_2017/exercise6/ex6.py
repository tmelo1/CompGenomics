
# coding: utf-8

# In[49]:

from __future__ import division
import numpy as np
from sklearn import linear_model
import sklearn.metrics as sk
import sys

f = open('genotype.csv')
snp_list = f.readline().strip().split(',')[1:9089]
f.close()
gen_data = np.genfromtxt('genotype.csv', delimiter=',', skip_header = 1)[:,1:]
allele_counts = []
maf = []
maf_gt03 = dict()
maf_gt05 = dict()
maf_gt10 = dict()
models = []
for row in gen_data.T:
    sum_0 = 0
    sum_1 = 0
    sum_2 = 0
    sum_3 = 0
    for i in row:
        if (i == 0.0):
            sum_0 += 1
        elif (i == 1.0):
            sum_1 += 1
        elif (i == 2.0):
            sum_2 += 1
        else:
            sum_3 += 1
        allele_counts.append({0: sum_0, 1: sum_1, 2: sum_2, 3: sum_3})

for i in range(0, 9088):
    num_A = allele_counts[i][0] + 0.5*allele_counts[i][1]
    num_B = allele_counts[i][2] + 0.5*allele_counts[i][1]
    total = allele_counts[i][0] + allele_counts[i][1] + allele_counts[i][2]
    if (num_A > num_B):
        maf.append((num_B) / total)
        if ((num_B / total) > 0.03):
            maf_gt03[i] = snp_list[i]
        if ((num_B / total) > 0.05):
            maf_gt05[i] = snp_list[i]
        elif ((num_B / total) > .10):
            maf_gt10[i] = snp_list[i]
    else:
        maf.append((num_A) / total)
        if ((num_A / total) > 0.03):
            maf_gt03[i] = snp_list[i]
        if ((num_A / total) > 0.05):
            maf_gt05[i] = snp_list[i]
        if ((num_A / total) > .10):
            maf_gt10[i] = snp_list[i]


for key in maf_gt05:
    trainX = np.genfromtxt('genotype.csv', delimiter=',', skip_header = 1, usecols = key)
    trainY = np.genfromtxt('phenotype.csv', delimiter=',', skip_header = 1, usecols = 1)
    model_dat = linear_model.LogisticRegression(C=1e86)
    model_dat.fit(trainX, trainY)
    models.append(model_dat)



# In[48]:

trainX.shape


# In[ ]:

trainY.shape


# In[ ]:



