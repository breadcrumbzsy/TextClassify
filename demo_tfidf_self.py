# -*- coding: utf-8 -*-

import os
import numpy as np
from sklearn import linear_model
import cPickle as Pickle

import LogisticRegression
from TextClassify import TFIDFforFiles
from TextClassify import TextClassify
from sklearn.externals import joblib

## 每次运行 训练集示例 训练集的文档数
input_path = 'data'
train_tfidf = TFIDFforFiles(os.path.join(input_path, 'train'))
N = train_tfidf.doc_num #train集有多少个txt
train_tfidf.N=N
##

## 初次运行,保存 统计的训练集的 train_tf和train_df
# train_tf, train_df = train_tfidf.compute_tfidf()
# train_tf, train_df = train_tfidf.reduce_tfidf(train_tf, train_df)
# print "=================="
# Pickle.dump(train_tf, open(os.path.join(input_path, 'train_tf.pkl'), 'wb'))
# print "saved train_tf.pkl"
# Pickle.dump(train_df, open(os.path.join(input_path, 'train_df.pkl'), 'wb'))
# print "saved train_df.pkl"
##

## 之后运行,加载 训练集的 train_tf和train_df
train_tf = Pickle.load(open(os.path.join(input_path, 'train_tf.pkl'), 'rb'))
print "loaded train_tf.pkl"
train_df = Pickle.load(open(os.path.join(input_path, 'train_df.pkl'), 'rb'))
train_tfidf.doc_freq=train_df
print "loaded train_df.pkl"

##

## 首次运行 保存 训练的LR模型
train_feature, train_target = train_tfidf.tfidf_feature(os.path.join(input_path, 'train'),train_tf, train_df, N)
# logreg = linear_model.LogisticRegression()
# logreg.fit(train_feature, train_target)
lr= LogisticRegression.LogisticRegression(train_feature, train_target, 1.0, 2000, regularized=False, normalization='l2')
learntParameters, final_costs = lr.train(train_feature, train_target, np.unique(train_target))
joblib.dump(learntParameters, 'lr_tfidf_self_learntParameters.model')
joblib.dump(final_costs, 'lr_tfidf_self_final_costs.model')
##

## 之后运行 加载 LR模型
learntParameters = joblib.load('lr_tfidf_self_learntParameters.model')
final_costs=joblib.load('lr_tfidf_self_final_costs.model')
print 'Number of classes', len(np.unique(train_target))
print 'learntParameters(one per class): '
for learntParameter in learntParameters:
	print learntParameter
print 'final_costs: '
print final_costs
##

## 测试运行 测试集==============!!!!!!!!!!!
test_tfidf = TFIDFforFiles(os.path.join(input_path, 'test'))
test_tf, test_df = test_tfidf.compute_tfidf()
test_tf, test_df = test_tfidf.reduce_tfidf(test_tf, test_df)
test_feature, test_target = test_tfidf.tfidf_feature(os.path.join(input_path, 'test'),test_tf, train_df, N)
## PREDICT
classifedLabels = []
for eachData in test_feature:
    classifedLabels.append(lr.classify(eachData, learntParameters))
classifedLabels = np.array(classifedLabels)
##

## ACCURACY
print 'Accuracy on training data: ', (np.sum(classifedLabels == test_target) / len(test_target)) * 100, '%'
##

## TextClassify
print "---------------------------------------------start test_single"
feature = train_tfidf.trainsorm_single_file(file)
pred = lr.classify(feature, learntParameters)
print pred
