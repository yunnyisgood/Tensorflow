import numpy as np
from sklearn.datasets import load_diabetes, load_boston, load_breast_cancer
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

'''
feature_engineering
=> PCA, pca.explained_variance_ratio_
'''

dataset = load_diabetes()
x = dataset.data
y = dataset.target

print(x.shape, y.shape)
# (506, 13) (506,)

pca = PCA(n_components=8) # 10 -> 7개로 컬럼을 압축 
# -> 일정비율로 압축. 컬럼을 삭제한다는 의미는 아님
# -> 차원을 축소해서 성능이 좋아질수도, 아닐수도 있음
x = pca.fit_transform(x)
print(x)
print(x.shape)
# (506, 7)

pca_EVR = pca.explained_variance_ratio_
print(pca_EVR)
# [0.40242142 0.14923182 0.12059623 0.09554764 0.06621856 0.06027192
#  0.05365605 0.04336832]

print(sum(pca_EVR))
# 0.9913119559917797

cumsum = np.cumsum(pca_EVR)
print(cumsum)
# [0.40242142 0.55165324 0.67224947 0.76779711 0.83401567 0.89428759
#  0.94794364 0.99131196]

print(np.argmax(cumsum >= 0.94)+1)
# 7 -> 0.94 정도의 합을 원한다면 n_components=7로 지정하면 된다

plt.plot(cumsum)
plt.grid()
plt.show()

'''x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=66, shuffle=True
)

# modeling
model = XGBRegressor()

# fit
model.fit(x_train, y_train)

# evaluate
results = model.score(x_test, y_test)
print('결과: ', results)'''



'''
압축 전
결과:  0.23802704693460175

압축 후
결과:  0.3366979304013662
'''