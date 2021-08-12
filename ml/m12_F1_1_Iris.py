'''
feautre_importances 가 전체 중요도에서 20%인 컬럼들을 제거하여 데이터셋을 재구성후 
# 각 모델별로 돌려서 결과 도출 
'''

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# data
datasets = load_iris()

df = pd.DataFrame(data=datasets.data, columns=datasets.feature_names)
df['target'] = datasets.target
print(df)

# x_train, x_test, y_train, y_test = train_test_split(
#     datasets.data, datasets.target, train_size=0.8, random_state=66
# )

# 컬럼삭제 
x_train, x_test, y_train, y_test = train_test_split(
    df.iloc[:, 1:], datasets.target, train_size=0.8, random_state=66
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
# [0.         0.0125026  0.03213177 0.95536562]
# 해당 컬럼의 acc에 대한 중요도를 주는지를 보여주는 지표 

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


DecisionTreeClassifier(max_depth=5)
>> 기존
acc:  0.9666666666666667
[0.         0.0125026  0.53835801 0.44913938]

>> 변경 후
컬럼 삭제 후 acc:  1.0
[0.         0.         0.50622624 0.49377376]

RandomForestClassifier
>> 기존
acc:  0.9333333333333333
[0.11372599 0.01977341 0.38207715 0.48442344]

컬럼 삭제 후 acc:  1.0
[0.00235577 0.22275534 0.36561362 0.40927527]

GradientBoostingClassifier
>> 기존
acc:  0.9666666666666667
[0.00283637 0.01223331 0.31112126 0.67380905]

>> 변경 후 
acc:  1.0
[3.28481035e-16 1.62064117e-01 1.12549537e-01 7.25386346e-01]

XGBClassifier
>> 기존
acc:  0.9
[0.01835513 0.0256969  0.62045246 0.3354955 ]

>> 변경 후 
acc:  1.0
[5.6700571e-03 2.2434165e-01 5.4559656e-05 7.6993370e-01]
'''