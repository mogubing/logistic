# -*- coding: utf-8 -*-
_author_ = 'huihui.gong'
_date_ = '2020/1/17'

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib as mpl
#防止中文乱码
mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False
cancerdata=pd.read_csv('D:\\PycharmProjects\\actualProject1116\\decisiontree200114\\datas\\breast-cancer-wisconsin.data',header=None,sep=',')
# 发现数据集中有个问号
cancerdata=cancerdata.replace('?',np.nan).dropna()
print(cancerdata.isnull().any())
print(cancerdata.shape)
cancerdata.drop_duplicates()
# print(cancerdata.shape)
x=cancerdata.iloc[:,1:10]
y=cancerdata.iloc[:,10]
# print(x.shape)
# print(y.shape)
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,random_state=10)
print(x_train)
print(x_train.dtypes)
# print(y_train)
cancerclf=LogisticRegressionCV(Cs=10,fit_intercept=True,penalty='l2',solver='lbfgs')
cancerclf.fit(x_train,y_train)
y_predict=cancerclf.predict(x_test)
print(cancerclf.score(x_test,y_test))
# 画图，对比预测值和实际值
plt.figure(figsize=(14,7))
x_len=len(x_test)
plt.plot(range(x_len),y_test,markersize=8,label=u'实际值',c='r',marker='o',zorder=3)
plt.plot(range(x_len),y_predict,markersize=14,label=u'预测值,R^2=%0.3f'%(cancerclf.score(x_test,y_test)),c='g',marker='o',zorder=2)
plt.ylim(0,6)
plt.ylabel(u'乳腺癌类型')
plt.xlabel(u'数据编号')
plt.title(u'logistic回归算法进行乳腺癌分类')
plt.legend()
plt.show()

