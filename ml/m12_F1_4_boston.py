'''
feautre_importances 가 전체 중요도에서 20%인 컬럼들을 제거하여 데이터셋을 재구성후 
# 각 모델별로 돌려서 결과 도출 
'''

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# data
datasets = load_boston()

df = pd.DataFrame(data=datasets.data, columns=datasets.feature_names)
df['target'] = datasets.target
print(df)
print(len(df.columns))
# 14

# x_train, x_test, y_train, y_test = train_test_split(
#     datasets.data, datasets.target, train_size=0.8, random_state=66
# )

# 컬럼삭제 
x_train, x_test, y_train, y_test = train_test_split(
    df.iloc[:, [ 2,  4,  5,  7,  8,  9, 10, 12]], datasets.target, train_size=0.8, random_state=66
)

# modeling
model = DecisionTreeRegressor()
# tree가 몇개의 층으로 되어있는가 -> max_depth
# max_depth=4 : 4개의 층으로 되어있는 구조

# model = RandomForestRegressor()

# model = GradientBoostingRegressor()

# model = XGBRegressor()


# fit
model.fit(x_train, y_train)

# evaluate
acc = model.score(x_test, y_test)
# print('acc: ', acc)
print('컬럼 삭제 후 acc: ', acc)
print(model.feature_importances_)


# -> 상위 20% 컬럼 추출 
# print(np.sort(model.feature_importances_)[::-1])
# print(np.where(model.feature_importances_> np.sort(model.feature_importances_)[::-1][8]))
# (array([ 2,  4,  5,  7,  8,  9, 10, 12], dtype=int64),)





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
acc:  0.8507309980875365
[3.72032030e-02 0.00000000e+00 0.00000000e+00 0.00000000e+00
 1.46473555e-02 2.90925176e-01 3.55772919e-04 5.93330743e-02
 0.00000000e+00 2.36939730e-02 0.00000000e+00 0.00000000e+00
 5.73841446e-01]
 컬럼 삭제 후 acc:  0.8315152418670131
[0.01006447 0.02579997 0.29664828 0.07734854 0.00393673 0.02886371
 0.01653378 0.54080453]

RandomForestRegressor
acc:  0.9204121088819914
[0.04041872 0.00099842 0.00640488 0.00114337 0.0235724  0.38563088
 0.01401355 0.06668028 0.00424658 0.01386344 0.01749286 0.01158781
 0.41394681]
 컬럼 삭제 후 acc:  0.9157971806775572
[0.00836335 0.0292714  0.38704793 0.08656205 0.00654201 0.01921034
 0.02585021 0.43715269]

 GradientBoostingRegressor
 acc:  0.9214527311223446
[0.07547322 0.00053953 0.36187798 0.01601405 0.02320444 0.04020009
 0.00989754 0.47279315]
 컬럼 삭제 후 acc:  0.9378807114181436
[0.00215413 0.05147327 0.35423126 0.09002422 0.00211846 0.01612157
 0.03783525 0.44604183]

XGBRegressor
acc:  0.8916904818896605
[0.03593979 0.00904454 0.32565036 0.01641634 0.03540455 0.04766432
 0.01550256 0.51437753]

 컬럼 삭제 후 acc:  0.923667949973626
[0.01225692 0.08588526 0.2620473  0.06029187 0.0271467  0.04528749
 0.09926033 0.40782416]
 '''