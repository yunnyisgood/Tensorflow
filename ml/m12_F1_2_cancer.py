'''
feautre_importances 가 전체 중요도에서 20%인 컬럼들을 제거하여 데이터셋을 재구성후 
# 각 모델별로 돌려서 결과 도출 
'''

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# data
datasets = load_breast_cancer()

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
    df.iloc[:, [ 0,  3,  6,  7, 20, 22, 23, 27]], datasets.target, train_size=0.8, random_state=66
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

'''
-> 컬럼당 acc의 평균 값보다 큰 컬럼들만 추출
print(np.mean(model.feature_importances_))
print(np.where(model.feature_importances_> np.mean(model.feature_importances_)))
(array([ 0,  3,  6,  7, 20, 22, 23, 27], dtype=int64),)
'''




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
acc:  0.9298245614035088
[0.         0.05940707 0.         0.00468454 0.00702681 0.
 0.         0.01967507 0.         0.00624605 0.         0.
 0.         0.01233852 0.         0.01405362 0.02248579 0.00902208
 0.         0.         0.         0.01612033 0.         0.71474329
 0.         0.         0.         0.11419683 0.         0.        ]

 컬럼 삭제 후 acc:  0.9210526315789473
[0.01232499 0.00456966 0.03383142 0.01495181 0.02218406 0.01093059
 0.75277255 0.14843492]

RandomForestClassifier
acc:  0.9824561403508771
[0.040175   0.01417973 0.02825478 0.05317835 0.00814518 0.00973534
 0.04448582 0.05600268 0.00385447 0.00364012 0.02479853 0.00492999
 0.01063662 0.03743225 0.00409775 0.00368067 0.00910304 0.00825301
 0.00446725 0.00446052 0.15021152 0.02338659 0.1119398  0.1656345
 0.01048854 0.01299796 0.02978691 0.10622578 0.01054816 0.0052691 ]

컬럼 삭제 후 acc:  0.9649122807017544
[0.04039674 0.07547801 0.04840778 0.12267697 0.15201065 0.21713661
 0.19217513 0.15171811]

 GradientBoostingClassifier
 acc:  0.9473684210526315
[5.99626091e-04 3.71744422e-02 2.69151350e-04 1.99419622e-03
 3.11046187e-03 3.67799287e-05 8.23083254e-04 1.38151427e-01
 3.26701236e-03 3.34477211e-04 4.02532511e-03 6.11420995e-06
 8.70810106e-04 1.78774586e-02 1.42559207e-03 3.81297566e-03
 1.07213607e-03 7.10743389e-04 1.62920716e-05 4.18658804e-04
 3.14356345e-01 4.19286889e-02 3.59809121e-02 2.71596526e-01
 3.36880420e-03 9.61129947e-05 1.35952295e-02 1.02509446e-01
 4.10038608e-05 5.30169138e-04]

컬럼 삭제 후 acc:  0.9649122807017544
[0.00841149 0.01408852 0.03517093 0.11749632 0.29104917 0.0979849
 0.28942952 0.14636915]

XGBClassifier
acc:  0.9736842105263158
[0.01420499 0.03333857 0.         0.02365488 0.00513449 0.06629944
 0.0054994  0.09745205 0.00340272 0.00369179 0.00769184 0.00281184
 0.01171023 0.0136856  0.00430626 0.0058475  0.00037145 0.00326043
 0.00639412 0.0050556  0.01813928 0.02285903 0.22248562 0.28493083
 0.00233393 0.         0.00903706 0.11586285 0.00278498 0.00775311]

 컬럼 삭제 후 acc:  0.9649122807017544
[0.02843658 0.01906539 0.03414661 0.10333769 0.06616882 0.26034892
 0.3595151  0.12898095]
'''