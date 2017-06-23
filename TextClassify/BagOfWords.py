# -*- coding: utf-8 -*-

import os
import re
import jieba
import numpy
import chardet
from scipy import sparse
from sklearn.externals import joblib

class BagOfWords:
    def __init__(self, dir):
        self.dir = dir
        
    def build_dictionary(self):
        dict_set = set()
        count = 0
        for (dirname, dirs, files) in os.walk(self.dir):
            for file in files:
                if file.endswith('.txt'):
                    filename = os.path.join(dirname, file)
                    with open(filename, 'rb') as f:
                        count += 1
                        for line in f:
                            line = self.process_line(line)
                            words = jieba.cut(line.strip(), cut_all=False)
                            dict_set |= set(words)
        self.num_samples = count
        # for w in dict_set:
        #     print w
        self.dict = self.reduce_dict(dict_set)
        
    def load_dictionary(self, dir):
        import cPickle as Pickle  
        try:
            print "loaded dictionary from %s" % dir
            self.dict = Pickle.load(open(dir, 'rb'))
            print "done"            
        except IOError:
            print "error while loading from %s" % dir
            
    def save_dictionary(self, dir):
        import cPickle as Pickle
        Pickle.dump(self.dict, open(dir, 'wb'))
        print "saved dictionary to %s" % dir
                
    def reduce_dict(self, dict_set):
        dict_copy = dict_set.copy()
        # 去掉单字/数字
        for word in dict_set:
            if len(word) < 2:
                dict_copy.remove(word)
            else:
                try:
                    float(word)
                    dict_copy.remove(word)
                except ValueError:
                    continue
        dictionary = {}
        for idx, word in enumerate(dict_copy):
            # print "idx="+str(idx)
            # print "word="+word
            # 以word为键,idx为值,如{'上海':0,'记者':1,'网络':2}   不过生成的字典是unicode的,看不出中文是啥
            dictionary[word] = idx
        # print dictionary
        return dictionary

    def process_line(self, line):
        line = line.decode("utf-8")
        return re.sub("]-·[\s+\.\!\/_,$%^*(+\"\':]+|[+——！，。？、~@#￥%……&*（）():\"=《]+".decode("utf8"),
                                           " ".decode("utf-8"), line)

    def transform_data(self, dir):
        from scipy import sparse
        print "transforming data in to bag of words vector"
        data = []
        target = []
        count = 0
        for (dirname, dirs, files) in os.walk(dir):
            for file in files:
                if file.endswith('.txt'):
                    count += 1
                    filename = os.path.join(dirname, file)
                    tags = re.split('[/\\\\]', dirname)
                    tag = tags[-1]
                    # print tag #互联网/等类文件夹名
                    # print str(len(self.dict))
                    word_vector = numpy.zeros(len(self.dict))
                    with open(filename, 'rb') as f:
                        for line in f:
                            line = self.process_line(line)
                            words = jieba.cut(line.strip(), cut_all=False)
                            for word in words:
                                try:
                                    word_vector[self.dict[word]] += 1
                                except KeyError:
                                    pass
                    #data.append(sparse.csr_matrix(word_vector))
                    data.append(word_vector)
                    target.append(tag)
        # self.num_samples = count
        print "done"
        # for d in data:
        #     print d
        # for t in target:
        #     print t
        m=numpy.asarray(data)
        # print m
        m_=sparse.csr_matrix(m)
        # print m_
        return m_,numpy.asarray(target)


    def trainsorm_single_file(self, file):
        word_vector = numpy.zeros(len(self.dict))
        with open(file, 'rb') as f:
            for line in f:
                line = self.process_line(line)
                words = jieba.cut(line.strip(), cut_all=False)
                for word in words:
                    try:
                        word_vector[self.dict[word]] += 1
                    except KeyError:
                        pass
        return word_vector


#以下为已经停用的无效方法,仅示纪念

    def build_train(self, data_dir):
        self.train_feature, self.train_target = self.transform_data(os.path.join(data_dir, 'train'))

    def save_train_feature(self, dir):
        import cPickle as Pickle
        Pickle.dump(self.train_feature, open(dir, 'wb'))
        print "saved train_feature to %s" % dir

    def load_train_feature(self, dir):
        import cPickle as Pickle
        try:
            print "loaded train_feature from %s" % dir
            self.train_feature = Pickle.load(open(dir, 'rb'))
            print "done"
        except IOError:
            print "error while loading from %s" % dir

    def save_train_target(self, dir):
        import cPickle as Pickle
        Pickle.dump(self.train_target, open(dir, 'wb'))
        print "saved train_target to %s" % dir

    def load_train_target(self, dir):
        import cPickle as Pickle
        try:
            print "loaded train_target from %s" % dir
            self.train_target = Pickle.load(open(dir, 'rb'))
            print "done"
        except IOError:
            print "error while loading from %s" % dir

    def train_result(self):
        return self.train_feature,self.train_target

