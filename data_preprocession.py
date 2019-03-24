#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 11:54:19 2019

@author: zhangzhaopeng
"""

import csv
phrase_douban = []
label_douban = []
with open("/Users/zhangzhaopeng/统计学习/机器学习/heihei/SpiderResult_renmin.csv", encoding='gbk') as file:
    douban = csv.reader(file)
    
    for row in douban:
        if row[3] == "Null":
            continue
        if len(row[1]) == 0:
            continue
        phrase_douban.append(row[1])
        label_douban.append(row[3])
        
label_douban.remove(label_douban[0])        
phrase_douban.remove(phrase_douban[0])