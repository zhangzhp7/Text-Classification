#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 14:11:47 2019

@author: zhangzhaopeng
"""

import pickle

## import data
data_path = '/Users/zhangzhaopeng/统计学习/机器学习/Text_Classification/data_preprocessing.pkl'
fp = open(data_path, 'rb')
x_train, x_test, y_train, y_test = pickle.load(fp)
fp.close()

## 卡方检验选择特征
from sklearn import naive_bayes
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
vectorizer = CountVectorizer(min_df = 2)
x_train_tf = vectorizer.fit_transform(x_train)
x_test_tf = vectorizer.transform(x_test)
chi2 = SelectKBest(chi2, k = 4000)
x_train_chi2 = chi2.fit_transform(x_train_tf, y_train)
x_test_chi2 = chi2.transform(x_test_tf)

## naive bayes
naive_chi2 = naive_bayes.MultinomialNB().fit(x_train_chi2, y_train)
naive_chi2_preds = naive_chi2.predict(x_test_chi2)
count_accu = 0
for i in range(len(y_test)):
    if y_test[i] == naive_chi2_preds[i]:
        count_accu += 1
naive_accu_chi2 = count_accu / len(y_test)
#naive_accu2 = metrics.accuracy_score(naive_preds, y_test)
print("Test set accuracy: ", naive_accu_chi2)
# confusion_matrix 
conf_arr_naive_chi2 = [[0, 0], [0, 0]]
for i in range(len(y_test)):
    if y_test[i] == 0:
        if naive_chi2_preds[i] == 0:
            conf_arr_naive_chi2[0][0] = conf_arr_naive_chi2[0][0] + 1
        else:
            conf_arr_naive_chi2[1][0] = conf_arr_naive_chi2[1][0] + 1
    elif y_test[i] == 1:
        if naive_chi2_preds[i] == 0:
            conf_arr_naive_chi2[0][1] = conf_arr_naive_chi2[0][1] + 1
        else :
            conf_arr_naive_chi2[1][1] = conf_arr_naive_chi2[1][1] + 1
recall_naive_chi2 = conf_arr_naive_chi2[0][0]/(conf_arr_naive_chi2[0][0] + conf_arr_naive_chi2[1][0]) 
print("recall: ", recall_naive_chi2)

## tf-idf选择特征
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=u'(?u)\\b[^\\d\\W]\\w+\\b', max_features=2000)
tfidf_vect.fit(x)
x_train_tfidf =  tfidf_vect.transform(x_train)
x_test_tfidf =  tfidf_vect.transform(x_test)    
word=tfidf_vect.get_feature_names() #获取词袋模型中的所有词语
weight=x_train_tfidf.toarray()

### naive bayes
naive = naive_bayes.MultinomialNB().fit(x_train_tfidf, y_train)
naive_preds = naive.predict(x_test_tfidf)
count_accu = 0
for i in range(len(y_test)):
    if y_test[i] == naive_preds[i]:
        count_accu += 1
naive_accu_tfidf = count_accu / len(y_test)
#naive_accu2 = metrics.accuracy_score(naive_preds, y_test)
print("Test accuracy: ", naive_accu_tfidf)
# confusion_matrix 
conf_arr_naive = [[0, 0], [0, 0]]
for i in range(len(y_test)):
    if y_test[i] == 0:
        if naive_preds[i] == 0:
            conf_arr_naive[0][0] = conf_arr_naive[0][0] + 1
        else:
            conf_arr_naive[1][0] = conf_arr_naive[1][0] + 1
    elif y_test[i] == 1:
        if naive_preds[i] == 0:
            conf_arr_naive[0][1] = conf_arr_naive[0][1] + 1
        else :
            conf_arr_naive[1][1] = conf_arr_naive[1][1] + 1
naive_recall_tfidf = conf_arr_naive[0][0]/(conf_arr_naive[0][0] + conf_arr_naive[1][0]) 
naive_accu3 = (conf_arr_naive[0][0] + conf_arr_naive[1][1]) / len(y_test)
print("召回率：", naive_recall_tfidf)













