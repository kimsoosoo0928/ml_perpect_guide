from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
dt_clf = DecisionTreeClassifier()
train_data = iris.data
train_label = iris.target

dt_clf.fit(train_data, train_label)

# 학습 데이터 세트로 예측 수행 
pred = dt_clf.predict(train_data)
print('예측 정확도: ',accuracy_score(train_label,pred))