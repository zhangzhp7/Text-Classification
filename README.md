# Text-Classification
- 使用FastText和LSTM, 在Keras框架下实现文本分类
- 对比方法：使用卡方统计量、tf-idf特征选择方法的朴素贝叶斯分类器  
# 数据描述
豆瓣上电视剧《人民的名义》的用户5万多条短评论文本及其评分（1-5星）。  
# 数据预处理
- 删除缺失值
- 合并类标签：1-3星为一类，4-5星为一类
- 中文分词
- 去标点和停用词
# 代码
data_preprocession.py 数据预处理  
lstm.py  LSTM方法实现文本分类  
fasttext.py  FastText方法  
features.py 特征选择：包括tf-idf,卡方统计量, LDA  
naive_bayes.py 朴素贝叶斯分类器  
baselines.py Logistic Regression  
# 评价指标
精确率、召回率
# 结果

# Todo
LDA
