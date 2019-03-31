import pandas as pd
from sklearn import model_selection
import numpy as np
import  matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
import jieba
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import jieba.analyse
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import numpy as np
import gensim
from sklearn.ensemble import AdaBoostClassifier
import datetime
from collections import defaultdict
starttime = datetime.datetime.now()

#long running



"""

 C:\\Users\比利王\Desktop\毕业设计\BosonNLP_sentiment_score
 clf = svm.SVC(C=2,kernel='poly',gamma=1).fit(x_train, y_train)

print ('test111:',clf.score(x_test, y_test))
ada = AdaBoostClassifier(DecisionTreeClassifier(min_samples_leaf=5),n_estimators=200, learning_rate=1)
ada.fit(x_train,y_train)
print('test:',ada.score(x_test,y_test))
print('trian:',ada.score(x_train,y_train))
test_r = ada.predict(x_test)

train_agg = pd.read_csv('F:\\pycharm\\workspace\\test_1\\train_x_y.csv',encoding='utf-8',sep=' ')

train_x = train_agg.iloc[:, 0:100]
train_y = train_agg['y']
x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.33, random_state=5)
"""
def c(x):
    if x >= 0:
        return int(0)
    else:
        return int(1)
def l2d(wordlist):
    data={}
    for x in range(len(wordlist)):
        data[wordlist[x]] = x
    return data

def conf(sendict,data,neg_word):
    w = 1
    score = 0

    senword = defaultdict()
    negword = defaultdict()
    for word in data.keys():
        if word in sendict.keys() and word not in neg_word:
            senword[data[word]] = sendict[word]
        elif word in neg_word :
            negword[data[word]] = -1
    sen_index_list = list(senword.keys())
    sen_index = 0
    for i in range(len(data.keys())):
        if sen_index < len(sen_index_list):
            if i not in senword.keys():
                for j in range(i,sen_index_list[sen_index]):
                    if j in neg_word :
                        w *= -1
                i = sen_index_list[sen_index]
            else:
                score += w*float(senword[i])
                sen_index +=1




    return score
if __name__ == '__main__':

    file_in = 'F:\\pycharm\\workspace\\test_1\\np_f2.txt'
    score = 0
    file_data = open(file_in,'r',encoding='utf-8')
    data = file_data.readlines()
    result_y = []
    sen_file = open('BosonNLP_sentiment_score.txt','r',encoding='utf-8')
    senlist = sen_file.readlines()
    sendict = defaultdict()
    for i in senlist:
        i.strip()
        if i == '\n':
            continue
        sendict[i.split(' ')[0]] = i.split(' ')[1]
    neg_file = open('neg.txt', 'r', encoding='utf-8')
    neg_word = neg_file.readlines()
    for i in data:
        #cut_dict.append(l2d(i.strip().split(' ')))
        data_dict = l2d(i.strip().split(' '))
        score=conf(sendict,data_dict,neg_word)
        if score>0:
            score=0
        else:
            score=1
        result_y.append(score)


    train = pd.read_csv('F:\\pycharm\\workspace\\test_1\\train_x_y.csv', sep=' ', encoding='utf-8')
    result_y_real = list(train['y'])
    k=0
    for i in range(len(result_y)):
        if result_y[i]==result_y_real[i]:
            k+=1
    print('正确率为：',precision_score(result_y_real,result_y))
    print('召回率为',recall_score(result_y_real,result_y))

    endtime = datetime.datetime.now()

    print (endtime - starttime)






