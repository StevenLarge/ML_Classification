#This is a multilabel classifier form MNIST data
#
#Steven Large
#April 30th 2018

import os
import matplotlib
import matplotlib.pyplot as plt
import scipy as sp

import numpy as np

from sklearn.datasets import fetch_mldata
from sklean.neighbors import KNeighborsClassifier


mnist = fetch_mldata("MNIST original")

[X,y] = mnist["data"],mnist["target"]

X_test,X_train,y_test,y_train = X[60000:], X[:60000], y[60000:], y[:60000]

shuffle_index = np.random.permutation(60000)

X_train,y_train = X_train[shuffle_index], y_train[shuffle_index]

y_train_large = (y_train >= 7)
y_train_odd = (ytrain % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)



