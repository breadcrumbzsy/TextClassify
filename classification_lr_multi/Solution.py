from __future__ import division

import numpy as np

import LogisticRegression

def sigmoidCalc(data):
    g = 1.0 / (1.0 + np.exp(-data))
    return g

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

## Multiclass or 3 class data classification
data, labels = readFile("classification_lr_multi/iris-data.txt", True)
m, n = data.shape
# lr1 = LogisticRegression.LogisticRegression(data, labels, 1.0, 8000, regularized=True, normalization = 'l1')
lr1 = LogisticRegression.LogisticRegression(data, labels, 1.0, 2000, regularized=True, normalization='l2')
learntParameters, final_costs = lr1.train(data, labels, np.unique(labels))
print 'Number of classes', len(np.unique(labels))
print 'learntParameters(one per class): '
for learntParameter in learntParameters:
	print learntParameter
print 'final_costs: '
print final_costs
# print len(learntParameters)
classifedLabels = []
for eachData in data:
    classifedLabels.append(lr1.classify(eachData, learntParameters))
classifedLabels = np.array(classifedLabels)

# print 'original label', 'classifedLabels'
# for each in zip(labels, classifedLabels):
# 	print each[0],', ', each[1],', ', each[0]==each[1]
print 'Accuracy on training data: ', (np.sum(classifedLabels == labels) / len(labels)) * 100, '%'


## 2 class data classification
# data, labels = readFile("classification_lr_multi/mod-iris.txt", True)
# m, n = data.shape
# lr1 = LogisticRegression.LogisticRegression(data, labels, 2.0, 100)
# learntParameters, final_costs = lr1.train(data, labels, np.unique(labels))
# print 'Number of classes', len(np.unique(labels))
# print 'learntParameters(only 1 learnt parameter): ', learntParameters
# print 'final_costs: ', final_costs
#
# classifedLabels = []
# for eachData in data:
#     classifedLabels.append(lr1.classify(eachData, learntParameters))
# classifedLabels = np.array(classifedLabels)
# # print 'original label', 'classifedLabels'
# # for each in zip(labels, classifedLabels):
# # 	print each[0],', ', each[1],', ', each[0]==each[1]
#
# print (np.sum(classifedLabels == labels) / len(labels)) * 100, '%'
