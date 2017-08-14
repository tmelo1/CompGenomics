
# coding: utf-8

# In[40]:

import numpy as np
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from __future__ import division
import sys

#argv order: train_expression, phen_train, phen_test, test_expression

trainX_all = np.genfromtxt(sys.argv[1], delimiter=',', skip_header = 1, unpack=True)[1:]
trainY = np.genfromtxt(sys.argv[2], delimiter=',', skip_header = 1, usecols = 1)

trainX_10 = trainX_all[:,:10]
testY = np.genfromtxt(sys.argv[3], delimiter=',', skip_header = 1, usecols = 1)

testX_all = np.genfromtxt(sys.argv[4], delimiter=',', skip_header = 1, unpack=True)[1:]
testX_10 = testX_all[:,:10]

model_all = linear_model.LogisticRegression(C=1e86)
model_all.fit(trainX_all, trainY)
model_10 = linear_model.LogisticRegression(C=1e86)
model_10.fit(trainX_10, trainY)

predictY_all = model_all.predict(testX_all)
predictY_10 = model_10.predict(testX_10)

confusion_matrix_all = confusion_matrix(testY, predictY_all)
confusion_matrix_10 = confusion_matrix(testY, predictY_10)

precision_all = confusion_matrix_all[0,0] / (confusion_matrix_all[0,0] + confusion_matrix_all[1, 0])
precision_10 = confusion_matrix_10[0,0] / (confusion_matrix_10[0,0] + confusion_matrix_10[1,0])

recall_all = confusion_matrix_all[0,0] / (confusion_matrix_all[0,0] + confusion_matrix_all[0,1])
recall_10 = confusion_matrix_10[0,0] / (confusion_matrix_10[0,0] + confusion_matrix_10[0,1])





