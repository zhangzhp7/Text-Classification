#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 13:52:53 2019

@author: zhangzhaopeng
"""

import pickle

## import data
data_path = '/Users/zhangzhaopeng/统计学习/机器学习/Text_Classification/data_preprocessing.pkl'
fp = open(data_path, 'rb')
x_train, x_test, y_train, y_test = pickle.load(fp)
fp.close()

## fasttext
from keras.preprocessing.text import Tokenizer

vocab_size=4000
embedding_dim = 50
# 输入词的最大长度
maxLen = len(max(x_train, key=len).split())

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(x_train)
X_train = tokenizer.texts_to_sequences(x_train)
X_test = tokenizer.texts_to_sequences(x_test)
    
from keras.preprocessing.sequence import pad_sequences
maxLen = len(max(x_train, key=len).split())
X_train = pad_sequences(X_train, padding='post', maxlen=maxLen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxLen)

from keras.layers import GlobalAveragePooling1D
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.embeddings import Embedding

model_ftt = Sequential()
#adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model_ftt.add(Embedding(vocab_size,
                    embedding_dim,
                    input_length=maxLen))
# GlobalAveragePooling1D 叠加平均输入的词向量
model_ftt.add(GlobalAveragePooling1D())
model_ftt.add(Dense(1, activation='sigmoid'))
model_ftt.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model_ftt.summary()
model_ftt.fit(X_train, y_train, epochs = 50, batch_size = 512, shuffle = True)

ftt_preds = model_ftt.predict(X_test)
#preds_fastt2 = preds_fastt.apply(lambda x: 1 if x >= 0.5 else 0)
#ftt_preds2 = (np.asarray(ftt_preds)).round()
import numpy as np
ftt_preds2 = np.zeros((len(ftt_preds),1))
for i in range(len(ftt_preds)):
    if ftt_preds[i] >= 0.5:
        ftt_preds2[i] = 1
conf_arr_ftt = [[0, 0], [0, 0]]
for i in range(len(y_test)):
    if y_test[i] == 0:
        if ftt_preds2[i] == 0:
            conf_arr_ftt[0][0] = conf_arr_ftt[0][0] + 1
        else:
            conf_arr_ftt[1][0] = conf_arr_ftt[1][0] + 1
    elif y_test[i] == 1:
        if ftt_preds2[i] == 0:
            conf_arr_ftt[0][1] = conf_arr_ftt[0][1] + 1
        else :
            conf_arr_ftt[1][1] = conf_arr_ftt[1][1] + 1
ftt_recall = conf_arr_ftt[0][0]/(conf_arr_ftt[0][0] + conf_arr_ftt[1][0])            

count_accu = 0
for i in range(len(y_test)):
    if y_test[i] == ftt_preds2[i]:
        count_accu += 1
ftt_accu = count_accu / len(y_test)   
print("Test accuracy: ", ftt_accu) 
print("召回率：", ftt_recall)      
#fastt_accu2 = metrics.accuracy_score(ftt_preds2, y_test)
#fastt_accu = metrics.accuracy_score(preds_fastt, y_test)
#loss, accu = model_ftt.evaluate(X_test, y_test)
#print("Test accuracy: ", accu)
#metrics.confusion_matrix(y_test, val_predict)