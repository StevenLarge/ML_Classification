#This is the driver routine for the linear Support Vector Machine (SVM) classifier
#
#Steven Large
#May 27th 2018

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

def GenerateScatterPlot(setosa_sl,setosa_sw,setosa_pl,setosa_pw,virginica_sl,virginica_sw,virginica_pl,virginica_pw,versicolor_sl,versicolor_sw,versicolor_pl,versicolor_pw):

	fig,ax = plt.subplots(1,2)

	ax[0].scatter(setosa_sl,setosa_sw,marker='o',color='b',label="Setosa")
	ax[0].scatter(virginica_sl,virginica_sw,marker='o',color='r',label="Virginica")
	ax[0].scatter(versicolor_sl,versicolor_sw,marker='o',color='g',label="Versicolor")

	ax[1].scatter(setosa_pl,setosa_pw,marker='o',color='b',label="Setosa")
	ax[1].scatter(virginica_pl,virginica_pw,marker='o',color='r',label="Virginica")
	ax[1].scatter(versicolor_pl,versicolor_pw,marker='o',color='g',label="Versicolor")

	ax[0].set_xlabel(r"Sepal length $cm$",fontsize=14)
	ax[0].set_ylabel(r"Sepal width $cm$",fontsize=14)

	ax[1].set_xlabel(r"Petal length $cm$", fontsize=14)
	ax[1].set_ylabel(r"Petal width $cm$", fontsize=14)

	ax[0].legend(loc='upper left', fontsize=12)
	ax[1].legend(loc='upper left', fontsize=12)

	plt.savefig("Plots/IrisScatterData.pdf",format="pdf")

	plt.show()
	plt.close()


#The iris dataset is for the classification of "Setosa" "Versicolor" and "Virginica", there are 50 instances of each type in the data set
#
#The array has two relevant data columns "data" and "target", the first containing the attribute vector [sepal length, sepal width, petal length, petal width]
#and the latter containig the target class 0-Setosa, 1-Versicolor, and 2-Virginica
#
#For information inlut the following command
#print iris["DESCR"]

iris = datasets.load_iris()

setosa_sepal_length = iris["data"][0:50, 0]
setosa_sepal_width = iris["data"][0:50, 1]
setosa_petal_length = iris["data"][0:50, 2]
setosa_petal_width = iris["data"][0:50, 3]

versicolor_sepal_length = iris["data"][51:100, 0]
versicolor_sepal_width = iris["data"][51:100, 1]
versicolor_petal_length = iris["data"][51:100, 2]
versicolor_petal_width = iris["data"][51:100, 3]

virginica_sepal_length = iris["data"][101:150, 0]
virginica_sepal_width = iris["data"][101:150, 1]
virginica_petal_length = iris["data"][101:150, 2]
virginica_petal_width = iris["data"][101:150, 3]

GenerateScatterPlot(setosa_sepal_length,setosa_sepal_width,setosa_petal_length,setosa_petal_width,
	virginica_sepal_length,virginica_sepal_width,virginica_petal_length,virginica_petal_width,
	versicolor_sepal_length,versicolor_sepal_width,versicolor_petal_length,versicolor_petal_width)

X = iris["data"][:, (2, 3)] 						# petal length, petal width
y = (iris["target"] == 2).astype(np.float64) 		# Iris-Virginica

svm_clf = Pipeline([
	("scaler", StandardScaler()),
	("linear_svc", LinearSVC(C=1, loss="hinge")),
	])

svm_clf.fit(X,y)

print svm_clf.predict([[5.5,1.7]])

