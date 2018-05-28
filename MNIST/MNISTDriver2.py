#this is the driver for the MNIST classificatio data
#
#Steven Large
#February 22nd 2018

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy as sp

import os

from sklearn.datasets import fetch_mldata
from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

from sklearn.neighbors import KNeighborsClassifier


def PlotNumberArray(X,Filename):

	fig,axes = plt.subplots(5,5,figsize=(4,4),sharex=True,sharey=True)

	axes[0,0].imshow(X[0].reshape(28,28),cmap = matplotlib.cm.binary, interpolation='nearest')
	axes[0,1].imshow(X[1].reshape(28,28),cmap = matplotlib.cm.binary, interpolation='nearest')
	axes[0,2].imshow(X[2].reshape(28,28),cmap = matplotlib.cm.binary, interpolation='nearest')
	axes[0,3].imshow(X[3].reshape(28,28),cmap = matplotlib.cm.binary, interpolation='nearest')
	axes[0,4].imshow(X[4].reshape(28,28),cmap = matplotlib.cm.binary, interpolation='nearest')

	axes[1,0].imshow(X[5].reshape(28,28),cmap = matplotlib.cm.binary, interpolation='nearest')
	axes[1,1].imshow(X[6].reshape(28,28),cmap = matplotlib.cm.binary, interpolation='nearest')
	axes[1,2].imshow(X[7].reshape(28,28),cmap = matplotlib.cm.binary, interpolation='nearest')
	axes[1,3].imshow(X[8].reshape(28,28),cmap = matplotlib.cm.binary, interpolation='nearest')
	axes[1,4].imshow(X[9].reshape(28,28),cmap = matplotlib.cm.binary, interpolation='nearest')

	axes[2,0].imshow(X[10].reshape(28,28),cmap = matplotlib.cm.binary, interpolation='nearest')
	axes[2,1].imshow(X[11].reshape(28,28),cmap = matplotlib.cm.binary, interpolation='nearest')
	axes[2,2].imshow(X[12].reshape(28,28),cmap = matplotlib.cm.binary, interpolation='nearest')
	axes[2,3].imshow(X[13].reshape(28,28),cmap = matplotlib.cm.binary, interpolation='nearest')
	axes[2,4].imshow(X[14].reshape(28,28),cmap = matplotlib.cm.binary, interpolation='nearest')

	axes[3,0].imshow(X[15].reshape(28,28),cmap = matplotlib.cm.binary, interpolation='nearest')
	axes[3,1].imshow(X[16].reshape(28,28),cmap = matplotlib.cm.binary, interpolation='nearest')
	axes[3,2].imshow(X[17].reshape(28,28),cmap = matplotlib.cm.binary, interpolation='nearest')
	axes[3,3].imshow(X[18].reshape(28,28),cmap = matplotlib.cm.binary, interpolation='nearest')
	axes[3,4].imshow(X[19].reshape(28,28),cmap = matplotlib.cm.binary, interpolation='nearest')

	axes[4,0].imshow(X[20].reshape(28,28),cmap = matplotlib.cm.binary, interpolation='nearest')
	axes[4,1].imshow(X[21].reshape(28,28),cmap = matplotlib.cm.binary, interpolation='nearest')
	axes[4,2].imshow(X[22].reshape(28,28),cmap = matplotlib.cm.binary, interpolation='nearest')
	axes[4,3].imshow(X[23].reshape(28,28),cmap = matplotlib.cm.binary, interpolation='nearest')
	axes[4,4].imshow(X[24].reshape(28,28),cmap = matplotlib.cm.binary, interpolation='nearest')

	axes[0,0].set_yticks([])
	axes[1,0].set_yticks([])
	axes[2,0].set_yticks([])
	axes[3,0].set_yticks([])

	axes[3,0].set_xticks([])
	axes[3,1].set_xticks([])
	axes[3,2].set_xticks([])		
	axes[3,3].set_xticks([])

	plt.savefig(Filename,format='pdf')
	plt.show()
	plt.close()


mnist = fetch_mldata("MNIST original")

[X,y] = mnist["data"],mnist["target"]

Digit = X[36000]

Digit = Digit.reshape(28,28)

plt.imshow(Digit, cmap = matplotlib.cm.binary, interpolation='nearest')
plt.axis('off')
plt.savefig("Plots/RandomDigit.pdf",format='pdf')
plt.show()
plt.close()

X_test,X_train,y_test,y_train = X[60000:], X[:60000], y[60000:], y[:60000]

shuffle_index = np.random.permutation(60000)

X_train,y_train = X_train[shuffle_index], y_train[shuffle_index]

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train,y_train_5)

cv_score_SGD_5 = cross_val_score(sgd_clf, X_train, y_train_5, cv=3)

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

ConfusionMatrix_SGD_5 = confusion_matrix(y_train_5,y_train_pred)

sgd_clf_MC = SGDClassifier(random_state=42)

sgd_clf_MC.fit(X_train,y_train)
sgd_clf_MC.predict(X[36000].reshape(1,-1))

y_train_pred = cross_val_predict(sgd_clf_MC, X_train, y_train, cv=3)

conf_mx = confusion_matrix(y_train,y_train_pred)

plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.savefig("Plots/ConfusionMatrix.pdf", format="pdf")
plt.show()
plt.close()

row_sums = conf_mx.sum(axis=1, keepdims=True)
#norm_conf_mx = conf_mx/row_sums
norm_conf_mx = np.true_divide(conf_mx,row_sums)
np.fill_diagonal(norm_conf_mx,0)


plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.savefig("Plots/ConfusionMatrix_Norm.pdf", format='pdf')
plt.show()
plt.close()

cl_a, cl_b = 3,5

X_aa = X_train[(y_train==cl_a)&(y_train_pred==cl_a)]
X_ab = X_train[(y_train==cl_a)&(y_train_pred==cl_b)]
X_ba = X_train[(y_train==cl_b)&(y_train_pred==cl_a)]
X_bb = X_train[(y_train==cl_b)&(y_train_pred==cl_b)]

PlotNumberArray(X_aa,"Plots/NumberMatrix_33.pdf")
PlotNumberArray(X_ab,"Plots/NumberMatrix_35.pdf")
PlotNumberArray(X_ba,"Plots/NumberMatrix_53.pdf")
PlotNumberArray(X_bb,"Plots/NumberMatrix_55.pdf")

y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large,y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train,y_multilabel)

print knn_clf.predict([X[36000]])


