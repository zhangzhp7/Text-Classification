#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 14:44:08 2019

@author: zhangzhaopeng
"""

import pickle

## import data
data_path = '/Users/zhangzhaopeng/统计学习/机器学习/Text_Classification/data_all.pkl'
fp = open(data_path, 'rb')
x, y, x_train, x_test, y_train, y_test = pickle.load(fp)
fp.close()

## tf-idf选择特征
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=u'(?u)\\b[^\\d\\W]\\w+\\b', max_features=2000)
tfidf_vect.fit(x)
x_train_tfidf =  tfidf_vect.transform(x_train)
x_test_tfidf =  tfidf_vect.transform(x_test)    
word=tfidf_vect.get_feature_names() #获取词袋模型中的所有词语
weight=x_train_tfidf.toarray()

## 卡方检验选择特征
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
vectorizer = CountVectorizer(min_df = 2)
x_train_tf = vectorizer.fit_transform(x_train)
x_test_tf = vectorizer.transform(x_test)
chi2 = SelectKBest(chi2, k = 4000)
x_train_chi2 = chi2.fit_transform(x_train_tf, y_train)
x_test_chi2 = chi2.transform(x_test_tf)

## LDA选择特征
from sklearn.decomposition import LatentDirichletAllocation
vectorizer=CountVectorizer(min_df=3, max_df=0.8)# 将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
vectorizer.fit(x)
x_train_tf = vectorizer.transform(x_train)
x_test_tf = vectorizer.transform(x_test)
lda = LatentDirichletAllocation(n_components=210)
x_train_lda = lda.fit_transform(x_train_tf)
x_test_lda = lda.transform(x_test_tf)

## store features
features = (x_train_tfidf, x_test_tfidf, x_train_chi2, x_test_chi2, x_train_lda, x_test_lda)       
fp = open('/Users/zhangzhaopeng/统计学习/机器学习/Text_Classification/features.pkl', 'wb')
pickle.dump(features, fp)
fp.close() 