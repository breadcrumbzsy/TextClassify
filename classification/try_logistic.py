# -*- coding: utf-8 -*-
import math
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))

dataSet=[[1,3,0,0,'0'],
         [1,3,0,1,'0'],
         [2,3,0,0,'1'],
         [3,2,0,0,'1'],
         [3,1,1,0,'1'],
         [3,1,1,1,'0'],
         [2,1,1,1,'1'],
         [1,2,0,0,'0'],
         [1,1,1,0,'1'],
         [3,2,1,0,'1'],
         [1,2,1,1,'1'],
         [2,2,0,1,'1'],
         [2,3,1,0,'1'],
         [3,2,0,1,'0']
         ]

dataSet=np.array(dataSet,float)
m,n=dataSet.shape
X=dataSet[:,0:n-1]
X = min_max_scaler.fit_transform(X)
Y=dataSet[:,-1]
print "==============X:"
print X
print "==============Y:"
print Y

X_train=X[0:m/2]
Y_train=Y[0:m/2]
X_test=X[m/2:m]
Y_test=Y[m/2:m]
print "==============X_train:"
print X_train
print "==============Y_train:"
print Y_train
print "==============X_test:"
print X_test
print "==============Y_test:"
print Y_test

def Sigmoid(z):
    G_of_Z = float(1.0 / float((1.0 + math.exp(-1.0*z))))
    return G_of_Z

def Hypothesis(theta, x):
    z = 0
    for i in xrange(len(theta)):
        z += x[i]*theta[i]
    print "z:",z
    print "h:",Sigmoid(z)
    return Sigmoid(z)

def Cost_Function(X,Y,theta,m):
    sumOfErrors = 0
    for i in xrange(m):
        xi = X[i]
        hi = Hypothesis(theta,xi)
        if Y[i] == 1:
            error = Y[i] * math.log(hi)
        elif Y[i] == 0:
            error = (1-Y[i]) * math.log(1-hi)
        print "error",error
        print "sumOfErrors:",sumOfErrors
        sumOfErrors += error
    const = -1/m
    J = const * sumOfErrors
    print 'cost is ', J
    return J

def Cost_Function_Derivative(X,Y,theta,j,m,alpha):
    sumErrors = 0
    for i in xrange(m):
        xi = X[i]
        xij = float(xi[j])
        hi = Hypothesis(theta,X[i])
        error = (hi - Y[i])*xij
        sumErrors += error
    m = len(Y)
    constant = float(alpha)/float(m)
    J = constant * sumErrors
    return J

def Gradient_Descent(X,Y,theta,m,alpha):
    new_theta = []
    for j in xrange(len(theta)):
        CFDerivative = Cost_Function_Derivative(X,Y,theta,j,m,alpha)
        new_theta_value = theta[j] - CFDerivative
        new_theta.append(new_theta_value)
    return new_theta

def Logistic_Regression(X,Y,alpha,theta,num_iters):
    m = len(X)
    for x in xrange(num_iters):
        new_theta = Gradient_Descent(X,Y,theta,m,alpha)
        theta = new_theta
        # if x % 100 == 0:
        #     Cost_Function(X,Y,theta,m)
        #     print 'theta ', theta
        #     print 'cost is ', Cost_Function(X,Y,theta,m)
    print "final theta:",theta
    Declare_Winner(theta)

def Declare_Winner(theta):
    countRight=0
    for i in xrange(len(X_test)):
        x=X_test[i]
        hi=round(Hypothesis(theta,x))
        if hi==Y_test[i]:
            print "对"
            countRight+=1
    myScore = float(countRight) / float(len(X_test))

    lr=LogisticRegression()
    lr.fit(X_train,Y_train)
    itsScore=lr.score(X_test,Y_test)

    print "myScore:",myScore
    print "itsScore:", itsScore



initTheta=[0,0]
alpha=0.1
num_iters=1000
Logistic_Regression(X_train,Y_train,alpha,initTheta,num_iters)