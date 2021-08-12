from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


# data
datasets = load_wine()
x_train, x_test, y_train, y_test = train_test_split(
    datasets.data, datasets.target, train_size=0.8, random_state=66
)

# modeling
model = DecisionTreeClassifier()
# tree가 몇개의 층으로 되어있는가 -> max_depth
# max_depth=4 : 4개의 층으로 되어있는 구조

# model = RandomForestClassifier()

# fit
model.fit(x_train, y_train)

# evaluate
acc = model.score(x_test, y_test)
print('acc: ', acc)

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
acc:  0.9444444444444444
[0.         0.00489447 0.         0.         0.01598859 0.
 0.18739896 0.         0.         0.04078249 0.0555874  0.33215293
 0.36319516]

RandomForestClassifier
acc:  1.0
[0.11595059 0.03439818 0.01590868 0.0355063  0.02917311 0.03377463
 0.16215738 0.01869469 0.01377743 0.15493632 0.08836895 0.12584763
 0.17150612]
'''