'''
Program to test LogisticRegression using Sklearn toolkit
'''
from __future__ import division

import numpy as np
from sklearn.linear_model import LogisticRegression

def readFile(filename, mode=False):
    f = open(filename, 'r')
    data = []
    labels = []
    for eachLine in f:
        eachLine = eachLine.strip()
        if (len(eachLine) == 0 or eachLine.startswith("#")):
            continue
        linedata = eachLine.split(",")
        # print linedata
        if (mode == True):
            data.append([1, float(linedata[0]), float(linedata[1]), float(linedata[2]), float(linedata[3])])
        else:
            data.append([float(linedata[0]), float(linedata[1]), float(linedata[2]), float(linedata[3])])
        labels.append(int(linedata[4]))
    data = np.array(data)
    labels = np.array(labels)
    return data, labels

data, labels = readFile("iris-data.txt", True)

lrclassfier = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=10.0, fit_intercept=True, intercept_scaling=1)
classifier = lrclassfier.fit(data, labels)
predicted_set = classifier.predict(data)

for i in zip(labels, predicted_set):
	print i[0],i[1],i[0]==i[1]

print 'Accuracy on training data: ',(np.sum(predicted_set == labels)/len(labels))*100,'%'