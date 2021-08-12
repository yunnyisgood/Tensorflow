'''
feautre_importances 가 전체 중요도에서 20%인 컬럼들을 제거하여 데이터셋을 재구성후 
# 각 모델별로 돌려서 결과 도출 
'''

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# data
datasets = load_diabetes()

df = pd.DataFrame(data=datasets.data, columns=datasets.feature_names)
df['target'] = datasets.target
print(df)
print(len(df.columns))
# 11

x_train, x_test, y_train, y_test = train_test_split(
    datasets.data, datasets.target, train_size=0.8, random_state=66
)

# # 컬럼삭제 
# x_train, x_test, y_train, y_test = train_test_split(
#     df.iloc[:, [ 2,  4,  5,  7,  8,  9, 10, 12]], datasets.target, train_size=0.8, random_state=66
# )

# modeling
# model = DecisionTreeRegressor()
# tree가 몇개의 층으로 되어있는가 -> max_depth
# max_depth=4 : 4개의 층으로 되어있는 구조

# model = RandomForestRegressor()

model = GradientBoostingRegressor()

# model = XGBRegressor()


# fit
model.fit(x_train, y_train)

# evaluate
acc = model.score(x_test, y_test)
print('acc: ', acc)
# print('컬럼 삭제 후 acc: ', acc)
print(model.feature_importances_)


# -> 상위 20% 컬럼 추출 
print(np.sort(model.feature_importances_)[::-1])
print(np.where(model.feature_importances_> np.sort(model.feature_importances_)[::-1][8]))
# (array([0, 2, 3, 4, 5, 6, 8, 9], dtype=int64),)





# visualize
# def plot_feature_importances_dataset(model):
#     n_features = datasets.data.shape[1]
#     plt.barh(np.arange(n_features), model.feature_importances_,
#     align='center')
#     plt.yticks(np.arange(n_features), datasets.feature_names)
#     plt.xlabel("Feature Importances")
#     plt.ylabel("Features")
#     plt.ylim(-1, n_features)

# plot_feature_importances_dataset(model)
# plt.show()



'''
DecisionTreeRegressor(max_depth=5)
acc:  -0.23408971940146617
[0.06556921 0.00798513 0.22343078 0.12004107 0.03890337 0.04268314
 0.04930075 0.01461038 0.36814456 0.06933161]

RandomForestRegressor
acc:  0.35604580883452863
[0.06561598 0.0098082  0.26704658 0.11997323 0.04247161 0.05185777
 0.04551607 0.02263238 0.30609331 0.06898487]

 GradientBoostingRegressor
 '''