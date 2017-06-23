# -*- coding: utf-8 -*-
import numpy as np
from scipy.sparse import csr_matrix
from collections import Counter

def kNN_Sparse(train_feature, train_target, test_feature, top_k):
    print("train_feature.shape:")
    print(train_feature.shape)
    print("test_feature.shape:")
    print(test_feature.shape)
    # calculate the square sum of each vector
    local_data_sq = train_feature.multiply(train_feature).sum(1)
    query_data_sq = test_feature.multiply(test_feature).sum(1)

    # calculate the dot
    distance = test_feature.dot(train_feature.transpose()).todense()

    # calculate the distance
    num_query, num_local = distance.shape
    print("distance.shape:")
    print(distance.shape)

    # print(np.tile(query_data_sq, (1, num_local)))
    # print("====")
    # print local_data_sq
    # print( np.tile(local_data_sq.T, (num_query, 1)))
    distance = np.tile(query_data_sq, (1, num_local)) + np.tile(local_data_sq.T, (num_query, 1)) - 2 * distance
    print distance
    # get the top k
    topK_idx = np.argsort(distance)[:, 0:top_k]
    # print(topK_idx)
    topK_similarity = np.zeros((num_query, top_k), np.float32)
    topK_classes=np.ndarray((num_query, top_k),basestring)
    print(xrange(num_query))
    for i in xrange(num_query):
        # print(i)
        # print(topK_idx[i])
        # print(distance[i, topK_idx[i]])
    	topK_similarity[i] = distance[i, topK_idx[i]]
        topK_classes[i]=train_target[topK_idx[i]]
    print("top-k")
    print(topK_similarity)
    print(topK_classes)
    return topK_similarity, topK_idx, topK_classes

def predict(train_feature, train_target, test_feature, top_k):
    topK_similarity, topK_idx, topK_classes=kNN_Sparse(train_feature, train_target, test_feature, top_k)
    num_query, top_k = topK_classes.shape
    result=[]
    for i in xrange(num_query):
        # print topK_classes[i]
        dic = Counter(topK_classes[i])
        top_one = dic.most_common(1)
        result.append(top_one[0][0])
    return result,topK_similarity, topK_idx, topK_classes



def run_knn():
    top_k = np.array(3, dtype=np.int32)
    local_data_offset = np.array([0, 1, 2, 4, 6], dtype=np.int64)
    local_data_index = np.array([0, 1, 0, 1, 0, 2], dtype=np.int32)
    local_data_value = np.array([1, 2, 3, 4, 8, 9], dtype=np.float32)
    local_data_csr = csr_matrix((local_data_value, local_data_index, local_data_offset), dtype = np.float32)
    print local_data_csr.todense()

    query_offset = np.array([0, 1, 4], dtype=np.int64)
    query_index = np.array([0, 0, 1, 2], dtype=np.int32)
    query_value = np.array([1.1, 3.1, 4, 0.1], dtype=np.float32)
    query_csr = csr_matrix((query_value, query_index, query_offset), dtype = np.float32)
    print query_csr.todense()

    target = [u'体育',u'政治',u'政治',u'd']
    topK_similarity, topK_idx,topK_classes = kNN_Sparse(local_data_csr,np.asarray(target), query_csr, top_k)

    for i in range(query_offset.shape[0]-1):
        print "for %d image, top %d is " % (i, top_k) , topK_idx[i]
        print "corresponding similarity: ", topK_similarity[i]
        print "corresponding class: ", str(topK_classes[i])

    print predict(topK_classes)

if __name__ == '__main__':
	run_knn()