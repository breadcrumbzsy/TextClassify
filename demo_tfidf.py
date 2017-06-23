# -*- coding: utf-8 -*-

import os
import numpy as np
from sklearn import linear_model
import cPickle as Pickle
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
logreg = linear_model.LogisticRegression()
logreg.fit(train_feature, train_target)
joblib.dump(logreg, 'lr_tfidf.model')
##

## 之后运行 加载 LR模型
logreg = joblib.load('lr_tfidf.model')
print logreg
##

## 测试运行 测试集==============!!!!!!!!!!!
test_tfidf = TFIDFforFiles(os.path.join(input_path, 'test'))
test_tf, test_df = test_tfidf.compute_tfidf()
test_tf, test_df = test_tfidf.reduce_tfidf(test_tf, test_df)
test_feature, test_target = test_tfidf.tfidf_feature(os.path.join(input_path, 'test'),test_tf, train_df, N)
## PREDICT
test_predict = logreg.predict(test_feature)
## ACCURACY
true_false = (test_predict==test_target)
accuracy = np.count_nonzero(true_false)/float(len(test_target))
print "accuracy is %f" % accuracy
##

## TextClassify
print "---------------------------------------------start test_single"
TextClassifier = TextClassify()
pred = TextClassifier.text_classify2('test.txt', train_tfidf, logreg)
print pred[0]


Pickle.dump(train_tf, open(os.path.join(input_path, 'train_tf.pkl'), 'wb'))
print "saved train_tf.pkl"
Pickle.dump(train_df, open(os.path.join(input_path, 'train_df.pkl'), 'wb'))
print "saved train_df.pkl"


train_tf = Pickle.load(open(os.path.join(input_path, 'train_tf.pkl'), 'rb'))
print "loaded train_tf.pkl"
train_df = Pickle.load(open(os.path.join(input_path, 'train_df.pkl'), 'rb'))
train_tfidf.doc_freq=train_df
print "loaded train_df.pkl"
