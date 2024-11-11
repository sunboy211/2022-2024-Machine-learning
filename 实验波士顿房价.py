from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.model_selection import LeaveOneOut
#boston=datasets.load_boston()
#boston_df = pd.DataFrame(boston.data, columns=boston.feature_names)
#boston_df['PRICE'] = boston.target  #增加一个 Price 属性作为目标值
#plt.figure(facecolor='gray')
#plt.scatter(boston_df['CRIM'], boston_df['PRICE'], s=30, edgecolor='white')
#plt.title('CRIM')
#plt.scatter(boston_df['LSTAT'], boston_df['PRICE'], s=30, edgecolor='white')
#plt.title('LSTAT')
#plt.scatter(boston_df['ZN'], boston_df['PRICE'], s=30, edgecolor='white')
#plt.title('ZN')
#plt.scatter(boston_df['INDUS'], boston_df['PRICE'], s=30, edgecolor='white')
#plt.title('INDUS')
#plt.scatter(boston_df['CHAS'], boston_df['PRICE'], s=30, edgecolor='white')
#plt.title('CHAS')
#plt.scatter(boston_df['NOX'], boston_df['PRICE'], s=30, edgecolor='white')
#plt.title('NOX')
#plt.scatter(boston_df['RM'], boston_df['PRICE'], s=30, edgecolor='white')
#plt.title('RM')
#plt.scatter(boston_df['AGE'], boston_df['PRICE'], s=30, edgecolor='white')
#plt.title('AGE')
#plt.scatter(boston_df['DIS'], boston_df['PRICE'], s=30, edgecolor='white')
#plt.title('DIS')
#plt.scatter(boston_df['RAD'], boston_df['PRICE'], s=30, edgecolor='white')
#plt.title('RAD')
#plt.scatter(boston_df['TAX'], boston_df['PRICE'], s=30, edgecolor='white')
#plt.title('TAX')
#plt.scatter(boston_df['PTRATIO'], boston_df['PRICE'], s=30, edgecolor='white')
#plt.title('PTRATIO')
#plt.scatter(boston_df['B'], boston_df['PRICE'], s=30, edgecolor='white')
#plt.title('B')
#plt.show()
'''
corrboston = boston_df.corr()
import seaborn as sns
sns.set()
plt.figure(figsize=(10,10))    #设置画布
sns.heatmap(corrboston,annot=True,cmap='RdGy')
plt.show()
'''
dataset = load_boston()
x_data = dataset.data # 导入所有特征变量
y_data = dataset.target # 导入目标值（房价）
name_data = dataset.feature_names #导入特征名
for i in range(13):
    #plt.subplot(7,2,i+1)
    plt.scatter(x_data[:,i],y_data,s = 20)
    plt.title(name_data[i])
    plt.show()

from sklearn.model_selection import train_test_split
#随机擦痒20%的数据构建测试样本，剩余作为训练样本
X_train,X_test,y_train,y_test=train_test_split(x_data,y_data,random_state=0,test_size=0.20)
print(len(X_train))
print(len(X_test))
print(len(y_train))
print(len(y_test))
#数据标归一化处理
from sklearn import preprocessing
#分别初始化对特征和目标值的标准化器
min_max_scaler = preprocessing.MinMaxScaler()
#分别对训练和测试数据的特征以及目标值进行标准化处理
X_train=min_max_scaler.fit_transform(X_train)
X_test=min_max_scaler.fit_transform(X_test)
y_train=min_max_scaler.fit_transform(y_train.reshape(-1,1))#reshape(-1,1)指将它转化为1列，行自动确定
y_test=min_max_scaler.fit_transform(y_test.reshape(-1,1))#reshape(-1,1)指将它转化为1列，行自动确定
lr=LinearRegression()
lrclf = LogisticRegression()
#使用训练数据进行参数估计
#lr.fit(X_train,y_train)
lrclf.fit(X_train,y_train)
#回归预测
#lr_y_predict=lr.predict(X_test)
lrclf_y_predict=lrclf.predict(X_test)
#score=cross_val_score(lr,y_test,lr_y_predict,cv=10)
#print('评分：', score)
score=cross_val_score(lrclf,y_test,lrclf_y_predict,cv=5) #cross_val_score(lrclf,y_test,lrclf_y_predict,scoring=?,cv=5)
print('评分：', score)

'''
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm

iris = datasets.load_iris()
print(iris.data.shape, iris.target.shape)

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
print(clf.score(X_test, y_test) )
'''