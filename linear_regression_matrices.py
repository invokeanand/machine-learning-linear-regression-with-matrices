#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 20:06:46 2020

@author: anandkadam
"""
import numpy as np
import matplotlib.pyplot as mp
import decimal as decimal


#X=np.array([[0],[1000],[1200],[800],[1500],[2000],[650]])

#Y=np.array([[0],[85],[86],[40],[110],[150],[25]])


from sklearn.datasets.samples_generator import make_regression
X, Y = make_regression(n_samples=200, n_features=1, n_informative=1, noise=6, bias=30, random_state=200)
m = 200


#Initialising Weights or M and C
W=np.zeros(shape=(1,2))




##INSERT X0 as first column with 1 as value.
X = np.insert(X, 0, 1, axis=1)

Y = np.reshape(Y, (200,1))
#print(np.shape(X))
#print(np.shape(W))
#print(np.shape(Y))
#Initialising number of iterations for learning
epochs = m
learning_rate = 0.02


i=0

#print(W[0][0])

for i in range(epochs):
    
    Y_hat = np.dot(X, np.transpose(W))
    
    #print(Y_hat)
    
    COST_J = np.subtract(Y_hat,Y)
    #print(np.shape(COST_J))
    #print(np.shape(Y_hat))
    #print(np.shape(Y))
    #print(np.size(W))
    j=0
    for j in range(np.size(W)):
        
        W[0][j] = W[0][j] - (learning_rate)*COST_J[i]*X[i][j]
        
 
print(W)

mp.plot(X[:,1],Y,'ro')
mp.plot([min(X[:,1]), max(X[:,1])], [min(Y_hat), max(Y_hat)], color='blue')

mp.show()







