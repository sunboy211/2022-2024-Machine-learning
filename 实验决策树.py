from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree
import pandas as pd
from sklearn import metrics

wine=load_wine()
X= wine.data
y = wine.target
pd.concat([pd.DataFrame(wine.data), pd.DataFrame(wine.target)], axis=1)
for i in range(13):
    plt.subplot(7,2,i+1)
    plt.scatter(X[ :,i],y ,s=20)
    plt.show()


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=0)
clf=tree.DecisionTreeClassifier(criterion='gini',splitter='random',max_depth=3,min_samples_leaf=10,min_samples_split=10)
#默认criterion='gini',splitter='best',max_depth=None,min_samples_split=2,min_samples_leaf=1,
clf = clf.fit(X_train,Y_train) #模型拟合
result = clf.score(X_test,Y_test)
print(result)
""""""
y_predict=clf.predict(X_test)
accuracy_score= metrics.accuracy_score(Y_test, y_predict)
recall_score = metrics.recall_score(Y_test, y_predict,average='micro')
print(accuracy_score,recall_score)