#coding=utf-8
def  createDataSet():
    dataSet=[[1,3,0,0,'no'],
             [1,3,0,1,'no'],
             [2,3,0,0,'yes'],
             [3,2,0,0,'yes'],
             [3,1,1,0,'yes'],
             [3,1,1,1,'no'],
             [2,1,1,1,'yes'],
             [1,2,0,0,'no'],
             [1,1,1,0,'yes'],
             [3,2,1,0,'yes'],
             [1,2,1,1,'yes'],
             [2,2,0,1,'yes'],
             [2,3,1,0,'yes'],
             [3,2,0,1,'no']
             ]
    labels = ['age','income',"student","credit_rating","buy_computer"]
    return dataSet, labels


def predict(vect,dataSet):
    countAll=len(dataSet)
    numFeature=len(vect)
    classList = [example[-1] for example in dataSet]
    uniqueClassList=set(classList)
    print(classList)
    subMap={}
    for className in uniqueClassList:
        subDataSet=[]
        subMap[className]=subDataSet
    for example in dataSet:
        length=len(example)
        subMap[example[-1]].append(example[0:length-1])
    print(subMap)
    probs={}
    bestProb=0
    bestClass=""
    for className in uniqueClassList:
        subDataSetOfThisClass=subMap[className]
        countThisClass=len(subDataSetOfThisClass)
        probThisClass=1.0*countThisClass/countAll
        probXwhenThisClass=1.0
        for i in range(len(vect)):
            featList = [example[i] for example in subDataSetOfThisClass]
            countAllOfclass=len(featList)
            countThisFeatureOfClass=0
            for j in featList:
                if j==vect[i]:
                    countThisFeatureOfClass+=1
            probXwhenThisClass*=(1.0*countThisFeatureOfClass/countAllOfclass)
        prob=probXwhenThisClass*probThisClass
        probs[className]=probXwhenThisClass*probThisClass
        if prob>bestProb:
            bestProb=prob
            bestClass=className
    return probs,bestProb,bestClass




def main():
    dataSet,label = createDataSet()
    vect=[1,2,1,0]
    probs,bestProb,bestClass=predict(vect,dataSet)
    print probs
    print bestProb
    print bestClass
if __name__=='__main__':
    main()