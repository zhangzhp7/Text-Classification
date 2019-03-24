#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 14:34:54 2019

@author: zhangzhaopeng
"""

import pickle

## import data
data_path = '/Users/zhangzhaopeng/统计学习/机器学习/Text_Classification/data_preprocessing.pkl'
fp = open(data_path, 'rb')
x_train, x_test, y_train, y_test = pickle.load(fp)
fp.close()

## import features
features_path = '/Users/zhangzhaopeng/统计学习/机器学习/Text_Classification/featrues.pkl'
fp = open(features_path, 'rb')
x_train_tfidf, x_test_tfidf, x_train_chi2, x_test_chi2, x_train_lda, x_test_lda = pickle.load(fp)
fp.close

### Logistic Regression
from sklearn.linear_model import LogisticRegression
## tf-idf选择特征
logistic_tfidf = LogisticRegression()
logistic_tfidf.fit(x_train_tfidf,y_train)
lr_preds = logistic_tfidf.predict(x_test_tfidf)
count_accu = 0
for i in range(len(y_test)):
    if y_test[i] == lr_preds[i]:
        count_accu += 1
lr_accu_tfidf = count_accu / len(y_test)
print("Test accuracy: ", lr_accu_tfidf)
conf_arr_lr = [[0, 0], [0, 0]]
for i in range(len(y_test)):
    if y_test[i] == 0:
        if lr_preds[i] == 0:
            conf_arr_lr[0][0] = conf_arr_lr[0][0] + 1
        else:
            conf_arr_lr[1][0] = conf_arr_lr[1][0] + 1
    elif y_test[i] == 1:
        if lr_preds[i] == 0:
            conf_arr_lr[0][1] = conf_arr_lr[0][1] + 1
        else :
            conf_arr_lr[1][1] = conf_arr_lr[1][1] + 1
lr_recall_tfidf = conf_arr_lr[0][0]/(conf_arr_lr[0][0] + conf_arr_lr[1][0]) 
print("召回率：", lr_recall_tfidf)

## 卡方统计量选择特征
logistic_chi2 = LogisticRegression()
logistic_chi2.fit(x_train_chi2,y_train)
lr_preds_chi2 = logistic_chi2.predict(x_test_chi2)
count_accu = 0
for i in range(len(y_test)):
    if y_test[i] == lr_preds_chi2[i]:
        count_accu += 1
lr_accu_chi2 = count_accu / len(y_test)
print("Test accuracy: ", lr_accu_chi2)
conf_arr_lr = [[0, 0], [0, 0]]
for i in range(len(y_test)):
    if y_test[i] == 0:
        if lr_preds_chi2[i] == 0:
            conf_arr_lr[0][0] = conf_arr_lr[0][0] + 1
        else:
            conf_arr_lr[1][0] = conf_arr_lr[1][0] + 1
    elif y_test[i] == 1:
        if lr_preds_chi2[i] == 0:
            conf_arr_lr[0][1] = conf_arr_lr[0][1] + 1
        else :
            conf_arr_lr[1][1] = conf_arr_lr[1][1] + 1
lr_recall_chi2 = conf_arr_lr[0][0]/(conf_arr_lr[0][0] + conf_arr_lr[1][0]) 
print("召回率：", lr_recall_chi2)










