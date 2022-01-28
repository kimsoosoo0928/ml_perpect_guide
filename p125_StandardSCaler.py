from sklearn.datasets import load_iris
import pandas as pd

# 붓꽃 데이터 세트를 로딩하고 DataFrame으로 변환합니다. 

iris = load_iris()
iris_data = iris.data
iris_df  = pd.DataFrame(data=iris_data, columns=iris.feature_names)

print('feature 들의 평균 값')
print(iris_df.mean())
print('\nfeature 들의 분산값')
print(iris_df.var())

# feature 들의 평균 값
# sepal length (cm)    5.843333
# sepal width (cm)     3.057333
# petal length (cm)    3.758000
# petal width (cm)     1.199333
# dtype: float64

# feature 들의 분산값
# sepal length (cm)    0.685694
# sepal width (cm)     0.189979
# petal length (cm)    3.116278
# petal width (cm)     0.581006
# dtype: float64

from sklearn.preprocessing import StandardScaler

# StandardScaler 객체 생성 
scaler = StandardScaler()
# StandardScaler로 데이터 세트 변환. fit()과 transform() 호출.
scaler.fit(iris_df)
iris_scaled = scaler.transform(iris_df)

# transform() 시 스케일 변환된 데이터 세트가 NumPy ndarray로 반환돼 이를 DataFrame으로 변환 
iris_df_scaled = pd.DataFrame(data=iris_scaled, columns=iris.feature_names)
print('feature 들의 평균 값')
print(iris_df_scaled.mean())
print('\nfeature 들의 분산 값')
print(iris_df_scaled.var())

# feature 들의 평균 값
# sepal length (cm)   -1.690315e-15
# sepal width (cm)    -1.842970e-15
# petal length (cm)   -1.698641e-15
# petal width (cm)    -1.409243e-15
# dtype: float64

# feature 들의 분산 값
# sepal length (cm)    1.006711
# sepal width (cm)     1.006711
# petal length (cm)    1.006711
# petal width (cm)     1.006711
# dtype: float64