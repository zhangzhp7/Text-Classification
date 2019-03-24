#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 12:40:57 2019

@author: zhangzhaopeng
"""

import pickle

## import data
data_path = '/Users/zhangzhaopeng/统计学习/机器学习/Text_Classification/data_preprocessing.pkl'
fp = open(data_path, 'rb')
x_train, x_test, y_train, y_test = pickle.load(fp)
fp.close()

### LSTM 

from keras.preprocessing.text import Tokenizer
# 特征词数
vocab_size=4000
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(x_train)
X_train = tokenizer.texts_to_sequences(x_train)
X_test = tokenizer.texts_to_sequences(x_test)
    
from keras.preprocessing.sequence import pad_sequences
maxLen = len(max(x_train, key=len).split())
X_train = pad_sequences(X_train, padding='post', maxlen=maxLen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxLen)

from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.layers import Bidirectional
embedding_dim = 50
embedding_layer = Embedding(vocab_size, embedding_dim, input_length = maxLen, trainable = True)
#print("weights[0][1][3] =", embedding_layer.get_weights().shape)
input_shape = (maxLen,)
sentence_indices = Input(input_shape, dtype='int32')
embeddings = embedding_layer(sentence_indices)
## 双向LSTM   
X = Bidirectional(LSTM(128, return_sequences = False))(embeddings)
X = Dropout(0.5)(X)

##单向LSTM
#X = LSTM(128, return_sequences = False)(X)
#X = Dropout(0.5)(X)

X = Dense(1)(X)
X = Activation("sigmoid")(X) 
model = Model(inputs = sentence_indices, outputs = X)
model.summary()
model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])

model.fit(X_train, y_train, epochs = 10, batch_size = 512, shuffle = True)
lstm_preds = model.predict(X_test)

import numpy as np
lstm_preds2 = np.zeros((len(lstm_preds),1))
for i in range(len(lstm_preds)):
    if lstm_preds[i] >= 0.5:
        lstm_preds2[i] = 1
#lstm_predict3 = (np.asarray(lstm_preds)).round()
# 混淆矩阵
conf_arr_lstm = [[0, 0], [0, 0]]
for i in range(len(y_test)):
    if y_test[i] == 0:
        if lstm_preds2[i] == 0:
            conf_arr_lstm[0][0] = conf_arr_lstm[0][0] + 1
        else:
            conf_arr_lstm[1][0] = conf_arr_lstm[1][0] + 1
    elif y_test[i] == 1:
        if lstm_preds2[i] == 0:
            conf_arr_lstm[0][1] = conf_arr_lstm[0][1] + 1
        else :
            conf_arr_lstm[1][1] = conf_arr_lstm[1][1] + 1
            
# 召回率
lstm_recall = conf_arr_lstm[0][0]/(conf_arr_lstm[0][0] + conf_arr_lstm[1][0]) 
# 精确率
count_accu = 0
for i in range(len(y_test)):
    if y_test[i] == lstm_preds2[i]:
        count_accu += 1
lstm_accu = count_accu / len(y_test)
print("Test accuracy: ", lstm_accu)
print("召回率：", lstm_recall)
#loss, accu = model.evaluate(X_test, y_test)
#print("Test accuracy: ", accu)
#metrics.confusion_matrix(y_test, lstm_predict)
#accuracy_test = (lstm_predict==y_test).mean()