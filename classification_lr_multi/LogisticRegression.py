# coding=utf-8
from __future__ import division
import numpy as np

class LogisticRegression:
    def __init__(self, data, labels, alpha = 1, num_iters = 100, regularized= False, debug = False, normalization = 'l2'):
        # 特征集,标签集,梯度下降的迭代次数,alpha的值
        self.normalization_mode = normalization
        self.regularized = regularized
        self.debug = debug
        self.num_iters = num_iters
        self.alpha = alpha


    def train(self, data, Olabels, unique_classes):
        print "train_feature:"
        print data
        print "train_lable/target:"
        print Olabels
        # 训练分类器,每一个标签一个分类器
        print 'training....'
        debug = self.debug
        regularized = self.regularized
        #print 'train regularized', regularized
        num_iters = self.num_iters
        m,n = data.shape
        # map labels to program friendly labels
        labels = np.zeros(Olabels.shape)

        uniq_Olabel_names = np.unique(Olabels)
        uniq_label_list = range(len(uniq_Olabel_names))
        for each in zip(uniq_Olabel_names, uniq_label_list):
            o_label_name = each[0]
            new_label_name = each[1]
            labels[np.where(Olabels == o_label_name)] = new_label_name

        labels = labels.reshape((len(labels),1))
        print unique_classes
        num_classes = len(unique_classes)
        Init_Thetas = [] # 存放theta的初始值
        Thetas = [] # 存放theta的最终值
        Cost_Thetas = [] # cost associated with each theta

        # if num_classes = 2, then N_Thetas will contain only 1 Theta
		# if num_classes >2, then N_Thetas will contain num_classes number of Thetas.

        if(num_classes == 2):
            theta_init = np.zeros((n,1))
            Init_Thetas.append(theta_init)
            # 只需要一个theta去分类A还是类B
            local_labels = labels
            assert(len(np.unique(labels)) == 2)
            assert(len(local_labels) == len(labels))
            init_theta = Init_Thetas[0]
            new_theta, final_cost = self.computeGradient(data, local_labels, init_theta)
            Thetas.append(new_theta)
            Cost_Thetas.append(final_cost)

        elif(num_classes>2):
            for eachInitTheta in range(num_classes):
                theta_init = np.zeros((n,1))
                Init_Thetas.append(theta_init)
                pass

            for eachClass in range(num_classes):
	 		    # load data local of the init_theta
	 			# +ve class is 1 and rest are zeros
	 			# its a one vs all classifier
                local_labels = np.zeros(labels.shape)
                local_labels[np.where(labels == eachClass)] = 1

                assert(len(np.unique(local_labels)) == 2)
                assert(len(local_labels) == len(labels))
                # print eachClass
                # print Init_Thetas
                init_theta = Init_Thetas[eachClass]
                new_theta, final_cost = self.computeGradient(data, local_labels, init_theta)
                #print final_cost
                Thetas.append(new_theta)
                Cost_Thetas.append(final_cost)
        return Thetas, Cost_Thetas


    def classify(self, data, Thetas):
        # since it is a one values all classifier, load all classifiers and pick most likely
        # i.e. which gives max value for sigmoid(X*theta)
        debug = self.debug
        assert(len(Thetas)>0)
        if(len(Thetas) > 1):
            mvals = []
            for eachTheta in Thetas:
                mvals.append(self.sigmoidCalc(np.dot(data, eachTheta)))
                pass
            return mvals.index(max(mvals))+1
        elif(len(Thetas) == 1):
            # 要么接近0,要么接近1. 小于0.5分为0,大于0.5分为1'
            print data
            print Thetas[0]
            print self.sigmoidCalc(np.dot(data, Thetas[0]))
            cval = round(self.sigmoidCalc(np.dot(data, Thetas[0])))+1.0
            #print 'classification output: ', cval
            return cval

    def sigmoidCalc(self, data):
        debug = self.debug
        data = np.array(data, dtype = np.longdouble)
        g = 1/(1+np.exp(-data))
        return g

    def computeCost(self,data, labels, init_theta):
        # 计算给定theta的代价
        debug = self.debug
        regularized = self.regularized
        if(regularized == True):
            llambda = 1.0
            #print 'using llambda', llambda
        else:
            llambda = 0
        m,n = data.shape
        J = 0
        grad = np.zeros(init_theta.shape)
        theta2 = init_theta[range(1,init_theta.shape[0]),:]
        if(self.normalization_mode == "l1"):
            regularized_parameter = np.dot(llambda/(2*m), np.sum( np.abs(theta2)))
            print 'mode: ', self.normalization_mode
            print 'lambda: ', llambda
            print regularized_parameter
        else:
            #(self.mode == "l2")
            regularized_parameter = np.dot(llambda/(2*m), np.sum( theta2 * theta2))
            print 'mode: ', self.normalization_mode
            print 'lambda: ', llambda
            print regularized_parameter
        J = (-1.0/ m) * ( np.sum( np.log(self.sigmoidCalc( np.dot(data, init_theta))) * labels + ( np.log ( 1 - self.sigmoidCalc(np.dot(data, init_theta)) ) * ( 1 - labels ) )))
        J = J + regularized_parameter
        # print 'llambda, regularized parameter: ', llambda, regularized_parameter
        return J

    def computeGradient(self,data, labels, init_theta):
        alpha = self.alpha
        debug = self.debug
        num_iters = self.num_iters
        m,n = data.shape
        regularized = self.regularized

        #print 'inoming regularized', regularized
        if(regularized == True):
            llambda = 1
        else:
            llambda = 0

        for eachIteration in range(num_iters):
            cost = self.computeCost(data, labels, init_theta)
            if(debug):
                print 'iteration: ', eachIteration
                print 'cost: ', cost
            #compute gradient
            B = self.sigmoidCalc(np.dot(data, init_theta) - labels)
            A = (1/m)*np.transpose(data)
            grad = np.dot(A,B)
            A = (self.sigmoidCalc(np.dot(data, init_theta)) - labels )
            B =  data[:,0].reshape((data.shape[0],1))
            grad[0] = (1/m) * np.sum(A*B)
            A = (self.sigmoidCalc(np.dot(data, init_theta)) - labels)
            B = (data[:,range(1,n)])
            for i in range(1, len(grad)):
                A = (self.sigmoidCalc(np.dot(data,init_theta)) - labels )
                B = (data[:,i].reshape((data[:,i].shape[0],1)))
                grad[i] = (1/m)*np.sum(A*B) + ((llambda/m)*init_theta[i])
            init_theta = init_theta - (np.dot((alpha/m), grad))
        return init_theta, cost


