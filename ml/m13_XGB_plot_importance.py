from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from xgboost.plotting import plot_importance
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

'''
plot_importance 예제
'''

# data
datasets = load_boston()
x_train, x_test, y_train, y_test = train_test_split(
    datasets.data, datasets.target, train_size=0.8, random_state=66
)

# modeling
# model = DecisionTreeRegressor(max_depth=5)
# tree가 몇개의 층으로 되어있는가 -> max_depth
# max_depth=4 : 4개의 층으로 되어있는 구조

# model = RandomForestRegressor()

model = XGBRegressor()

# fit
model.fit(x_train, y_train)

# evaluate
acc = model.score(x_test, y_test)
print('acc: ', acc)

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

plot_importance(model) # XGBRegressor를 모델로 사용했을 때만 사용가능 
plt.show()


'''
DecisionTreeRegressor(max_depth=5)
acc:  0.8507309980875365
[3.72032030e-02 0.00000000e+00 0.00000000e+00 0.00000000e+00
 1.46473555e-02 2.90925176e-01 3.55772919e-04 5.93330743e-02
 0.00000000e+00 2.36939730e-02 0.00000000e+00 0.00000000e+00
 5.73841446e-01]

RandomForestRegressor
acc:  0.9204121088819914
[0.04041872 0.00099842 0.00640488 0.00114337 0.0235724  0.38563088
 0.01401355 0.06668028 0.00424658 0.01386344 0.01749286 0.01158781
 0.41394681]
'''