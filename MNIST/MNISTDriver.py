#This is the python driver for the MNIST training data classifier problem
#
#Steven Large
#November 30th 2017

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

import os

from sklearn.datasets import fetch_mldata
from sklearn.linear_model import SGDClassifier


#----------------------- Plotting Routines --------------------------

def PlotImage(DigitData, Filename="Image.pdf", Show=False):

	DigitImage = DigitData.reshape(28,28)

	plt.imshow(DigitImage, cmap=matplotlib.cm.binary, interpolation="nearest")
	plt.axis("off")
	plt.savefig(Filename)

	if Show == True:
		plt.show()

	plt.close()





#---------------------- Main Driving Routines ------------------------


mnist = fetch_mldata('MNIST original')

[X,y] = mnist["data"],mnist["target"]

Digit1 = X[36000]
Digit2 = X[69000]
Digit3 = X[15000]

PlotImage(Digit1,"Image1.pdf")
PlotImage(Digit2,"Image2.pdf")
PlotImage(Digit3,"Image3.pdf")

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

y_train_1 = (y_train == 1)
y_train_2 = (y_train == 2)
y_train_3 = (y_train == 3)
y_train_4 = (y_train == 4)
y_train_5 = (y_train == 5)
y_train_6 = (y_train == 6)
y_train_7 = (y_train == 7)
y_train_8 = (y_train == 8)
y_train_9 = (y_train == 9)

y_test_1 = (y_test == 1)
y_test_2 = (y_test == 2)
y_test_3 = (y_test == 3)
y_test_4 = (y_test == 4)
y_test_5 = (y_test == 5)
y_test_6 = (y_test == 6)
y_test_7 = (y_test == 7)
y_test_8 = (y_test == 8)
y_test_9 = (y_test == 9)

sgd_clf1 = SGDClassifier(random_state=42)
sgd_clf1.fit(X_train,y_train_1)

sgd_clf2 = SGDClassifier(random_state=42)
sgd_clf2.fit(X_train,y_train_2)

sgd_clf3 = SGDClassifier(random_state=42)
sgd_clf3.fit(X_train,y_train_3)

sgd_clf4 = SGDClassifier(random_state=42)
sgd_clf4.fit(X_train,y_train_4)

sgd_clf5 = SGDClassifier(random_state=42)
sgd_clf5.fit(X_train,y_train_5)

sgd_clf6 = SGDClassifier(random_state=42)
sgd_clf6.fit(X_train,y_train_6)

sgd_clf7 = SGDClassifier(random_state=42)
sgd_clf7.fit(X_train,y_train_7)

sgd_clf8 = SGDClassifier(random_state=42)
sgd_clf8.fit(X_train,y_train_8)

sgd_clf9 = SGDClassifier(random_state=42)
sgd_clf9.fit(X_train,y_train_9)





