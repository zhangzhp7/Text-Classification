#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 23:48:14 2019

@author: zhangzhaopeng
"""

import pickle

## import all data
data_path = '/Users/zhangzhaopeng/统计学习/机器学习/Text_Classification/data_all.pkl'
fp = open(data_path, 'rb')
x, y, x_train, x_test, y_train, y_test = pickle.load(fp)
fp.close()

## 特征提取
from keras.preprocessing.text import Tokenizer
vocab_size=4000 # 特征词数
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(x_train)
X_train = tokenizer.texts_to_sequences(x_train)
X_test = tokenizer.texts_to_sequences(x_test)
    
from keras.preprocessing.sequence import pad_sequences
maxLen = len(max(x_train, key=len).split())
X_train = pad_sequences(X_train, padding='post', maxlen=maxLen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxLen)


from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Sequential

model = Sequential()
embedding_dim = 50
model.add(Embedding(vocab_size, embedding_dim, input_length=maxLen))
model.add(Dropout(0.2))
model.add(Conv1D(250, 3, padding='valid', activation='relu', strides=1))
model.add(MaxPooling1D(3))
model.add(Flatten())
model.add(Dense(embedding_dim, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])
model.fit(X_train, y_train, epochs = 10, batch_size = 512, shuffle = True)
#print model.evaluate(x_test, y_test)
cnn_preds = model.predict(X_test)

import numpy as np
cnn_preds2 = np.zeros((len(cnn_preds),1))
for i in range(len(cnn_preds)):
    if cnn_preds[i] >= 0.5:
        cnn_preds2[i] = 1
#lstm_predict3 = (np.asarray(lstm_preds)).round()
# 混淆矩阵
conf_arr_cnn = [[0, 0], [0, 0]]
for i in range(len(y_test)):
    if y_test[i] == 0:
        if cnn_preds2[i] == 0:
            conf_arr_cnn[0][0] = conf_arr_cnn[0][0] + 1
        else:
            conf_arr_cnn[1][0] = conf_arr_cnn[1][0] + 1
    elif y_test[i] == 1:
        if cnn_preds2[i] == 0:
            conf_arr_cnn[0][1] = conf_arr_cnn[0][1] + 1
        else :
            conf_arr_cnn[1][1] = conf_arr_cnn[1][1] + 1
            
# 召回率
cnn_recall = conf_arr_cnn[0][0]/(conf_arr_cnn[0][0] + conf_arr_cnn[1][0]) 
# 精确率
count_accu = 0
for i in range(len(y_test)):
    if y_test[i] == cnn_preds2[i]:
        count_accu += 1
cnn_accu = count_accu / len(y_test)
print("Test accuracy: ", cnn_accu)
print("召回率：", cnn_recall)








