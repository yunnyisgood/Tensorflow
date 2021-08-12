'''
feautre_importances 가 전체 중요도에서 20%인 컬럼들을 제거하여 데이터셋을 재구성후 
# 각 모델별로 돌려서 결과 도출 
'''

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# data
datasets = load_wine()

df = pd.DataFrame(data=datasets.data, columns=datasets.feature_names)
df['target'] = datasets.target
print(df)
print(len(df.columns))
# 31 -> target 제외 30 x 0. 8 = 24

# x_train, x_test, y_train, y_test = train_test_split(
#     datasets.data, datasets.target, train_size=0.8, random_state=66
# )

# 컬럼삭제 
x_train, x_test, y_train, y_test = train_test_split(
    df.iloc[:, [ 0,  3,  5,  6,  9, 10, 11, 12]], datasets.target, train_size=0.8, random_state=66
)

# modeling
model = DecisionTreeClassifier()
# tree가 몇개의 층으로 되어있는가 -> max_depth
# max_depth=4 : 4개의 층으로 되어있는 구조

# model = RandomForestClassifier()

# model = GradientBoostingClassifier()

# model = XGBClassifier()


# fit
model.fit(x_train, y_train)

# evaluate
acc = model.score(x_test, y_test)
# print('acc: ', acc)
print('컬럼 삭제 후 acc: ', acc)
print(model.feature_importances_)


# -> 상위 30% 컬럼 추출 
# print(np.sort(model.feature_importances_)[::-1])
# print(np.where(model.feature_importances_> np.sort(model.feature_importances_)[::-1][8]))
# (array([ 0,  3,  5,  6,  9, 10, 11, 12], dtype=int64),)





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
DecisionTreeClassifier
acc:  0.9444444444444444
[0.         0.00489447 0.         0.         0.01598859 0.
 0.18739896 0.         0.         0.04078249 0.0555874  0.33215293
 0.36319516]
컬럼 삭제 후 acc:  0.9444444444444444
[0.00489447 0.01598859 0.         0.18739896 0.04078249 0.0555874
 0.33215293 0.36319516]

RandomForestClassifier
acc:  1.0
[0.11595059 0.03439818 0.01590868 0.0355063  0.02917311 0.03377463
 0.16215738 0.01869469 0.01377743 0.15493632 0.08836895 0.12584763
 0.17150612]
 컬럼 삭제 후 acc:  1.0
[0.12857854 0.04586117 0.05325711 0.16531923 0.18231204 0.09527465
 0.13742072 0.19197653]

GradientBoostingClassifier
acc:  0.9722222222222222
[1.41654193e-02 4.19856744e-02 2.22575472e-02 3.45264619e-03
 3.74084934e-03 2.02398452e-05 1.08573832e-01 1.34363016e-04
 1.88762106e-04 2.50922898e-01 2.96292141e-02 2.48044218e-01
 2.76884337e-01]
컬럼 삭제 후 acc:  1.0
[0.05347645 0.00915791 0.00076713 0.11336084 0.24929201 0.03724527
 0.25952293 0.27717746]

 XGBClassifier
 acc:  1.0
[0.01854127 0.04139536 0.01352911 0.01686821 0.02422602 0.00758254
 0.10707161 0.01631111 0.00051476 0.12775211 0.01918284 0.50344414
 0.10358089]
컬럼 삭제 후 acc:  1.0
[0.02674782 0.02218918 0.01034164 0.1288798  0.1589059  0.02640564
 0.5090987  0.11743139]
'''