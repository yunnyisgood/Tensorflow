from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# data
datasets = load_breast_cancer()
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
# print('acc: ', acc)
print('컬럼 삭제 후 acc: ', acc)

print(model.feature_importances_)
# [0.         0.0125026  0.03213177 0.95536562]
# 해당 컬럼의 acc에 대한 중요도를 주는지를 보여주는 지표 

# visualize
def plot_feature_importances_dataset(model):
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
    align='center')
    plt.yticks(np.arange(n_features), datasets.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)
    
plot_feature_importances_dataset(model)
plt.show()


'''
DecisionTreeClassifier(max_depth=5)
acc:  0.9298245614035088
[0.         0.05940707 0.         0.00468454 0.00702681 0.
 0.         0.01967507 0.         0.00624605 0.         0.
 0.         0.01233852 0.         0.01405362 0.02248579 0.00902208
 0.         0.         0.         0.01612033 0.         0.71474329
 0.         0.         0.         0.11419683 0.         0.        ]

RandomForestClassifier
acc:  0.9824561403508771
[0.040175   0.01417973 0.02825478 0.05317835 0.00814518 0.00973534
 0.04448582 0.05600268 0.00385447 0.00364012 0.02479853 0.00492999
 0.01063662 0.03743225 0.00409775 0.00368067 0.00910304 0.00825301
 0.00446725 0.00446052 0.15021152 0.02338659 0.1119398  0.1656345
 0.01048854 0.01299796 0.02978691 0.10622578 0.01054816 0.0052691 ]
'''