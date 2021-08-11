import numpy as np
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score


datasets = load_breast_cancer() # (569, 30)

print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target

n_splits = 5
kfold = KFold(n_splits=n_splits,  shuffle=True, random_state=66)

#2.modeling
# model = LinearSVC()
# acc [0.86842105 0.92105263 0.92105263 0.92982456 0.96460177]
# 평균  acc:  0.921

# model = SVC()
# acc [0.89473684 0.92982456 0.89473684 0.92105263 0.96460177]
# 평균  acc:  0.921

# model =  KNeighborsClassifier()
# acc [0.92105263 0.92105263 0.92105263 0.92105263 0.95575221]
# 평균  acc:  0.928

# model = LogisticRegression()
# acc [0.93859649 0.95614035 0.88596491 0.94736842 0.96460177]
# 평균  acc:  0.9385

# model = DecisionTreeClassifier()
# acc [0.9122807  0.92105263 0.92105263 0.87719298 0.95575221]
# 평균  acc:  0.9175

model = RandomForestClassifier()
# acc [0.97368421 0.96491228 0.95614035 0.95614035 0.98230088]
# 평균  acc:  0.9666

#3. training

#4.predict
scores = cross_val_score(model, x, y, cv=kfold)
# 한번에 fit, evaluate까지
print('acc',scores)
print('평균  acc: ',round(np.mean(scores), 4))