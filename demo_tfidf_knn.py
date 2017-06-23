# -*- coding: utf-8 -*-

import os
import numpy as np
from sklearn import linear_model
import cPickle as Pickle
from TextClassify import TFIDFforFiles
from TextClassify import TextClassify
from sklearn.externals import joblib
import test_knn
from scipy import sparse


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

## 得到训练集的 feature & target
train_feature, train_target = train_tfidf.tfidf_feature(os.path.join(input_path, 'train'),train_tf, train_df, N)
# Pickle.dump(train_tf, open(os.path.join(input_path, 'train_feature.pkl'), 'wb'))
# Pickle.dump(train_tf, open(os.path.join(input_path, 'train_target.pkl'), 'wb'))
# print "save train_feature.pkl & train_target.pkl"
# train_target = Pickle.load(open(os.path.join(input_path, 'train_target.pkl'), 'rb'))
# train_feature_text=Pickle.load(open(os.path.join(input_path, 'train_feature.pkl'), 'rb'))
# train_feature = sparse.csr_matrix(np.asarray(train_feature_text))

print "loaded train_feature.pkl & train_target.pkl"
## 得到测试集的 feature & target
test_tfidf = TFIDFforFiles(os.path.join(input_path, 'input'))
test_tf, test_df = test_tfidf.compute_tfidf()
test_tf, test_df = test_tfidf.reduce_tfidf(test_tf, test_df)
test_feature, test_target = test_tfidf.tfidf_feature(os.path.join(input_path, 'input'),test_tf, train_df, N)
print(test_feature)

## PREDICT
k=60
test_result_60,topK_similarity, topK_idx, topK_classes=test_knn.predict(train_feature,train_target, test_feature, k)
k=50
test_result_50,topK_similarity, topK_idx, topK_classes=test_knn.predict(train_feature,train_target, test_feature, k)
k=40
test_result_40,topK_similarity, topK_idx, topK_classes=test_knn.predict(train_feature,train_target, test_feature, k)
k=30
test_result_30,topK_similarity, topK_idx, topK_classes=test_knn.predict(train_feature,train_target, test_feature, k)
k=20
test_result_20,topK_similarity, topK_idx, topK_classes=test_knn.predict(train_feature,train_target, test_feature, k)
k=10
test_result_10,topK_similarity, topK_idx, topK_classes=test_knn.predict(train_feature,train_target, test_feature, k)
k=5
test_result_5,topK_similarity, topK_idx, topK_classes=test_knn.predict(train_feature,train_target, test_feature, k)
##
for i in test_target:
    print i,
print " "

for i in test_result_60:
    print i,
print " "
print np.count_nonzero(test_result_60==test_target)/float(len(test_target))

for i in test_result_50:
    print i,
print " "
print np.count_nonzero(test_result_50==test_target)/float(len(test_target))

for i in test_result_40:
    print i,
print " "
print np.count_nonzero(test_result_40==test_target)/float(len(test_target))

for i in test_result_30:
    print i,
print " "
print np.count_nonzero(test_result_30==test_target)/float(len(test_target))

for i in test_result_20:
    print i,
print " "
print np.count_nonzero(test_result_20==test_target)/float(len(test_target))

for i in test_result_10:
    print i,
print " "
print np.count_nonzero(test_result_10==test_target)/float(len(test_target))

for i in test_result_5:
    print i,
print " "
print np.count_nonzero(test_result_5==test_target)/float(len(test_target))

## ACCURACY
# true_false = (test_result==test_target)
# accuracy = np.count_nonzero(true_false)/float(len(test_target))
# print "accuracy is %f" % accuracy
##



