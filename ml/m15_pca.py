import numpy as np
from sklearn.datasets import load_diabetes, load_boston, load_breast_cancer
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

print(x.shape, y.shape)
# (506, 13) (506,)

pca = PCA(n_components=12) # 10 -> 7개로 컬럼을 압축
x = pca.fit_transform(x)
print(x)
print(x.shape)
# (506, 7)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=66, shuffle=True
)

# modeling
model = XGBRegressor()

# fit
model.fit(x_train, y_train)

# evaluate
results = model.score(x_test, y_test)
print('결과: ', results)



'''
boston

압축 전
결과:  0.9221188601856797

PCA(n_components=7)
결과:  0.8874143051779056


'''