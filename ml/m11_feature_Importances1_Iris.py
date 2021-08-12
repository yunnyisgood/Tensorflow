from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# data
datasets = load_iris()
x_train, x_test, y_train, y_test = train_test_split(
    datasets.data, datasets.target, train_size=0.8, random_state=66
)

# modeling
# model = DecisionTreeClassifier()
# tree가 몇개의 층으로 되어있는가 -> max_depth
# max_depth=4 : 4개의 층으로 되어있는 구조

# model = RandomForestClassifier()

# model = GradientBoostingClassifier()

model = XGBClassifier()

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



'''
DecisionTreeClassifier
acc:  1.0
[0.         0.         0.50622624 0.49377376]

RandomForestClassifier
acc:  0.9333333333333333
[0.09267926 0.0256812  0.39666416 0.48497538]
'''