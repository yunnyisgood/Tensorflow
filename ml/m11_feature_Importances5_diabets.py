from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np



# data
datasets = load_diabetes()
x_train, x_test, y_train, y_test = train_test_split(
    datasets.data, datasets.target, train_size=0.8, random_state=66
)

# modeling
model = DecisionTreeRegressor()
# tree가 몇개의 층으로 되어있는가 -> max_depth
# max_depth=4 : 4개의 층으로 되어있는 구조

# model = RandomForestRegressor()

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
DecisionTreeRegressor(max_depth=5)
acc:  -0.23408971940146617
[0.06556921 0.00798513 0.22343078 0.12004107 0.03890337 0.04268314
 0.04930075 0.01461038 0.36814456 0.06933161]

RandomForestRegressor
acc:  0.35604580883452863
[0.06561598 0.0098082  0.26704658 0.11997323 0.04247161 0.05185777
 0.04551607 0.02263238 0.30609331 0.06898487]
'''