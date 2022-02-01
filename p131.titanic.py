from pyexpat import features
from argon2 import Parameters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

titanic_df = pd.read_csv('./data/titanic_train.csv')
print(titanic_df.head(3))

print('\n ### 학습 데이터 정보 ### \n')
print(titanic_df.info())


titanic_df['Age'].fillna(titanic_df['Age'].mean(), inplace=True)
titanic_df['Cabin'].fillna('N', inplace=True)
titanic_df['Embarked'].fillna('N', inplace=True)
print('데이터 세트 Null 값 개수', titanic_df.isnull().sum().sum())

print(' Sex 값 분포 :\n', titanic_df['Sex'].value_counts())
print('\n Cabin 값 분포 :\n', titanic_df['Cabin'].value_counts())
print('\n Embarked 값 분포 :\n', titanic_df['Embarked'].value_counts())

titanic_df['Cabin'] = titanic_df['Cabin'].str[:1]
print(titanic_df['Cabin'].head(3))

titanic_df.groupby(['Sex','Survived'])['Survived'].count()

# 입력 age에 따라 구분 값을 반환하는 함수 설정. DataFrame의 apply lamde 식에 사용.

def get_category(age):
    cat = ''
    if age <= -1: cat = 'unknown'
    elif age <= 5: cat= 'Baby'
    elif age <= 12: cat= 'Child'
    elif age <= 18: cat= 'Teenager'
    elif age <= 25: cat= 'Student'
    elif age <= 35: cat= 'Young Adult'
    elif age <= 60: cat= 'Adult'
    else : cat = 'Elderly'

    return cat

plt.figure(figsize=(10,6))

group_names = {'Unknown', 'Baby', 'Child', 'Teenager','Student', 'Young Adult', 'Adult', 'Elderly'}

titanic_df['Age_cat'] = titanic_df['Age'].apply(lambda x : get_category(x))

def encode_features(dataDF):
    features = ['Cabin','Sex', 'Embarked']
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(dataDF[feature])
        dataDF[feature] = le.transform(dataDF[feature])

    return dataDF

titanic_df = encode_features(titanic_df)

print(titanic_df.head())

# Null 처리 함수 

def fillna(df):
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['Cabin'].fillna('N', inplace=True)
    df['Embarked'].fillna('N', inplace=True)
    df['Fare'].fillna(0, inplace=True)
    return df

# 머신러닝 알고리즘에 불필요한 속성 제거
def drop_features(df):
    df.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True)
    return df

# 레이블 인코딩 수행

def format_features(df):
    df['Cabin'] = df['Cabin'].str[:1]
    features = ['Cabin', 'Sex', 'Embarked']
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df[feature])
        df[feature] = le.transform(df[feature])
    return df

# 앞에서 설정한 데이터 전처리 함수 호출
def transform_features(df):
    df = fillna(df)
    df = drop_features(df)
    df = format_features(df)
    return df 

# 원본 데이터를 재로딩하고, 피처 데이터 세트와 레이블 데이터 세트 추출
titanic_df = pd.read_csv('./data/titanic_train.csv')
y_titanic_df = titanic_df['Survived']
X_titanic_df = titanic_df.drop('Survived', axis=1)

X_titanic_df = transform_features(X_titanic_df)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_titanic_df, y_titanic_df,test_size=0.2,random_state=11)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

dt_clf = DecisionTreeClassifier(random_state=11)
rf_clf = RandomForestClassifier(random_state=11)
lr_clf = LogisticRegression()

# dt 
dt_clf.fit(X_train, y_train)
dt_pred = dt_clf.predict(X_test)
print('dt 정확도 : {0:.4f}'.format(accuracy_score(y_test, dt_pred)))

# rf
rf_clf.fit(X_train, y_train)
rf_pred = rf_clf.predict(X_test)
print('rf 정확도 : {0:.4f}'.format(accuracy_score(y_test, rf_pred)))

# lr
lr_clf.fit(X_train, y_train)
lr_pred = lr_clf.predict(X_test)
print('lr 정확도 : {0:.4f}'.format(accuracy_score(y_test, lr_pred)))

from sklearn.model_selection import cross_val_score

scores = cross_val_score(dt_clf, X_titanic_df, y_titanic_df, cv=5)
for iter_count, accuracy in enumerate(scores):
    print("교차 검증{0} 정확도 : {1:.4f}".format(iter_count, accuracy))

print("평균 정확도 :{0:.4f}".format(np.mean(scores)))

from sklearn.model_selection import GridSearchCV

parameters = {'max_depth':[2,3,5,10],'min_samples_split':[2,3,5], 'min_samples_leaf':[1,5,8]}

grid_dclf = GridSearchCV(dt_clf, param_grid=parameters, scoring='accuracy',cv=5)
grid_dclf.fit(X_train, y_train)

print('GridServhCV 최적 하이퍼 파라미터 :', grid_dclf.best_params_)
print('GridSearchCV 최고 정확도 : {0:.4f}'.format(grid_dclf.best_score_))
best_dclf = grid_dclf.best_estimator_

# GridSerchCV의 최적 하이퍼 파라미터로 학습된 Estimator로 예측 및 평가 수행.
dpredictions = best_dclf.predict(X_test)
accuracy = accuracy_score(y_test, dpredictions)
print('테스트 세트에서의 DT 정확도 : {0:4f}'.format(accuracy))