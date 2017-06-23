# -*- encoding:utf-8 -*-
from pyteaser import  Summarize,keywords, keywords5
from pprint import pprint
import json
from aylienapiclient import textapi



def singleTxt(txtname):
    code=0
    message="success"
    title=""
    time=""
    text=""

    f2=open(txtname)
    i=0
    while 1:
        line=f2.readline()
        if not line:
           break
        if i==0:
            title=line
        if i==1:
            time=line
        if i>=2:
            text=text+line
        i=i+1

    if i<2:
        code=1
        message="wrong format"
    print(text)
    key2=keywords5(text)
    pprint(key2)
    summaries=Summarize(title, text)
    pprint(summaries)
    abstract=''
    for summary in summaries:
        abstract=abstract+summary+" "

    print abstract

    client = textapi.Client("1808166e", "16c6e9275d7a517192b517abe12c9b35")
    sentimentstr = client.Sentiment({'text': text})
    sentiment=sentimentstr['polarity_confidence']
    print sentiment

    data={
        'code':code,
        'message':message,
        'title':title.strip('\n'),
        'time':time.strip('\n'),
        'abstract':abstract,
        'keywords':key2,
        'sentiment':sentiment
    }
    return data

txtname1="n1.txt"
txtname2="n2.txt"
data1=singleTxt(txtname1)
data2=singleTxt(txtname2)
articles={}
articles[txtname1]=data1
articles[txtname2]=data2
json_str=json.dumps(articles)
print json_str