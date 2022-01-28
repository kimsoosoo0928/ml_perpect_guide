from sklearn.datasets import load_iris
import pandas as pd

# 붓꽃 데이터 세트를 로딩하고 DataFrame으로 변환합니다. 

iris = load_iris()
iris_data = iris.data
iris_df  = pd.DataFrame(data=iris_data, columns=iris.feature_names)

from sklearn.preprocessing import MinMaxScaler

# MinMaxScaler 객체 생성 
scaler = MinMaxScaler()
# MinMaxScaler로 데이터 세트 변환. fit()과 transform() 호출.
scaler.fit(iris_df)
iris_scaled = scaler.transform(iris_df)

# transform() 시 스케일 변환된 데이터 세트가 NumPy ndarray로 반환돼 이를 DataFrame으로 변환 
iris_df_scaled = pd.DataFrame(data=iris_scaled, columns=iris.feature_names)
print('feature 들의 평균 값')
print(iris_df_scaled.mean())
print('\nfeature 들의 분산 값')
print(iris_df_scaled.var())


# feature 들의 평균 값
# sepal length (cm)    0.428704
# sepal width (cm)     0.440556
# petal length (cm)    0.467458
# petal width (cm)     0.458056
# dtype: float64

# feature 들의 분산 값
# sepal length (cm)    0.052908
# sepal width (cm)     0.032983
# petal length (cm)    0.089522
# petal width (cm)     0.100869
# dtype: float64