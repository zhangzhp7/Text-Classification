#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 11:54:19 2019

@author: zhangzhaopeng
"""
## import raw data
import csv
phrase_douban = []
label_douban = []
with open("/Users/zhangzhaopeng/统计学习/机器学习/Text_Classification/SpiderResult_renmin.csv", encoding='gbk') as file:
    douban = csv.reader(file)
    next(douban, None)
    
    for row in douban:
        if row[3] == "Null":
            continue
        if len(row[1]) == 0:
            continue
        phrase_douban.append(row[1])
        label_douban.append(row[3])

## 合并类标签
for i in range(len(label_douban)):
    if label_douban[i] in ["1", "2", "3"]:
        label_douban[i] = 0
    if label_douban[i] in ["4", "5"]:
        label_douban[i] = 1
## import stopwords        
stopwords = []
with open("/Users/zhangzhaopeng/统计学习/机器学习/Text_Classification/stopwords.txt", "r",encoding = "gbk") as f:
    for line in f.readlines():
        stopwords.append(line.strip())

## 分词，去掉停用词
import jieba
phrase_jieba = []
for line in phrase_douban:
    phrase_jieba.append(jieba.lcut(line))
phrase_nostop = []
for x in phrase_jieba:
    line = []
    for y in x:
        if y not in stopwords:
            line.append(y)
    phrase_nostop.append(line)

## 去标点    
import re         
def del_digit(line):
    punctuation = """！？｡,，。.＂＃＄％＆＇（）＊＋－／：；＜＝＞＠［＼］＾＿｀｛｜｝～《》｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…...‧﹏"""
    re_punctuation = "[{}]+".format(punctuation)
    line = re.sub(re_punctuation, "", line)
    return line.strip()

for i in range(len(phrase_nostop)):
    for j in range(len(phrase_nostop[i])):
        phrase_nostop[i][j] = del_digit(phrase_nostop[i][j])
for i in range(len(phrase_nostop)):
    while('' in phrase_nostop[i]):
        phrase_nostop[i].remove('')
        continue
    
### extract the nonempty phrase and its label
x = []
y = []
for i in range(len(phrase_nostop)):
    if len(phrase_nostop[i]) != 0:
        x.append(' '.join(phrase_nostop[i]))
        y.append(label_douban[i])
        
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state=1)

## store the data after preprocessing
import pickle
data = (x_train, x_test, y_train, y_test)
fp = open('/Users/zhangzhaopeng/统计学习/机器学习/Text_Classification/data_preprocessing.pkl', 'wb')
pickle.dump(data, fp)
fp.close()        
        
        
        
        
        
        
        
