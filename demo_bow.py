# -*- coding: utf-8 -*-

import os
import numpy as np
from sklearn import linear_model
from sklearn.externals import joblib

from TextClassify import BagOfWords
from TextClassify import TextClassify

data_dir = 'data'
## BAG OF WORDS MODEL
BOW = BagOfWords(os.path.join(data_dir, 'train'))
# BOW.build_dictionary()
# BOW.save_dictionary(os.path.join(data_dir, 'dicitionary.pkl'))
BOW.load_dictionary(os.path.join(data_dir, 'dicitionary.pkl'))
print "---------------------------------------------load_dictionary"

## LOAD DATA
# train_feature, train_target = BOW.transform_data(os.path.join(data_dir, 'train'))
# test_feature, test_target = BOW.transform_data(os.path.join(data_dir, 'test'))
# print "---------------------------------------------transform_data_test"

## TRAIN LR MODEL
# logreg = linear_model.LogisticRegression(C=1e5)
# logreg.fit(train_feature, train_target)
# joblib.dump(logreg, 'lr.model')
logreg = joblib.load('lr.model')
print logreg

## PREDICT
# test_predict = logreg.predict(test_feature) #[ 互联网  互联网  旅游  旅游 ...]
# print "---------------------------------------------predict_test_feature"

## ACCURACY
# true_false = (test_predict==test_target) #[ True  False  True  True ...]
# accuracy = np.count_nonzero(true_false)/float(len(test_target))
# print "accuracy is %f" % accuracy

## TextClassify
print "---------------------------------------------start test_single"
TextClassifier = TextClassify()
pred = TextClassifier.text_classify('test.txt', BOW, logreg)
print pred[0]

##需要改进的地方:
#1,换个算法,词加权的那种,TFIDF,SVM,理解一下LR(accuracy=0.828798)
#2,报了个小异常,归一化处理?:DeprecationWarning: Passing 1d arrays as data is deprecated in 0.17 and will raise ValueError in 0.19. Reshape your data either using X.reshape(-1, 1) if your data has a single feature or X.reshape(1, -1) if it contains a single sample.DeprecationWarning)
#3,使用全部数据集(目前只使用了一般数据),优化精简字典(根据权重,停用词等删减)是一定要在256000之内么