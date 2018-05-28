#This driver driver is for a nonlinear SVM classifier of Iris data 
#
#Steven Large
#May 27th 2018

import numpy as np
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

moons = make_moons()

X = moons[0]
y = moons[1]

polynomial_svm_clf = Pipeline([
	("poly_features", PolynomialFeatures(degree=3)),
	("scaler", StandardScaler()),
	("svm_clf", LinearSVC(C=10, loss="hinge"))
	])

polynomial_svm_clf.fit(X, y)


