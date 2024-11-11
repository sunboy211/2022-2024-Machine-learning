from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import neural_network
from sklearn import metrics
cancer=load_breast_cancer()
X=cancer.data
y=cancer.target
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)
clf=neural_network.MLPClassifier(hidden_layer_sizes=(50,50),solver='sgd',activation='logistic',max_iter=200)
'''
1.hidden_layer_sizes:传入一个元组， 例如（50，50），这表示第一层隐藏层和第二层隐藏层都有50个神经元。
2.solver: 权重优化器，{'lbfgs','sgd','adam'}, 默认是adam;
lbfgs：quasi-Newton方法的优化器;sgd：随机梯度下降;adam：机遇随机梯度的优化器
3.activation: 激活函数，{'identity','logistic','tanh','relu'}，默认是relu
'''
clf=clf.fit(X_train,Y_train)

y_predict=clf.predict(X_test)
accuracy_score= metrics.accuracy_score(Y_test, y_predict)
recall_score = metrics.recall_score(Y_test, y_predict,average='micro')
f1_score=metrics.f1_score(Y_test, y_predict)
print(accuracy_score,recall_score,f1_score)
