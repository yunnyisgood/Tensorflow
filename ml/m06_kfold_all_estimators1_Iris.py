import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.utils import all_estimators
from sklearn.model_selection import KFold, cross_val_score
import warnings
warnings.filterwarnings('ignore')

datasets = load_iris()

x = datasets.data
y = datasets.target

#2.modeling
allAlgorithms = all_estimators(type_filter='classifier')

print(allAlgorithms)
# [('AdaBoostClassifier', <class 'sklearn.ensemble._weight_boosting.AdaBoostClassifier'>), 이런식으로 출력됨 
print(len(allAlgorithms))
# 모델의 개수는 41개 

kfold = KFold(n_splits=5, shuffle=True, random_state=66)

for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()

        scores = cross_val_score(model, x, y, cv=kfold)

        print(name,'의 acc',scores)
        print(name,'의 평균  acc: ',round(np.mean(scores), 4))

    except:
        print(name,'은 없다 ')
        continue
    
'''
AdaBoostClassifier 의 acc [0.63333333 0.93333333 1.         0.9        0.96666667]
AdaBoostClassifier 의 평균  acc:  0.8867
BaggingClassifier 의 acc [0.93333333 0.96666667 1.         0.9        0.96666667]
BaggingClassifier 의 평균  acc:  0.9533
BernoulliNB 의 acc [0.3        0.33333333 0.3        0.23333333 0.3       ]
BernoulliNB 의 평균  acc:  0.2933
CalibratedClassifierCV 의 acc [0.9        0.83333333 1.         0.86666667 0.96666667]
CalibratedClassifierCV 의 평균  acc:  0.9133
CategoricalNB 의 acc [0.9        0.93333333 0.93333333 0.9        1.        ]
CategoricalNB 의 평균  acc:  0.9333
ClassifierChain 은 없다
ComplementNB 의 acc [0.66666667 0.66666667 0.7        0.6        0.7       ]
ComplementNB 의 평균  acc:  0.6667
DecisionTreeClassifier 의 acc [0.93333333 0.96666667 1.         0.9        0.93333333]
DecisionTreeClassifier 의 평균  acc:  0.9467
DummyClassifier 의 acc [0.3        0.33333333 0.3        0.23333333 0.3       ]
DummyClassifier 의 평균  acc:  0.2933
ExtraTreeClassifier 의 acc [0.9        1.         1.         0.86666667 0.96666667]
ExtraTreeClassifier 의 평균  acc:  0.9467
ExtraTreesClassifier 의 acc [0.93333333 0.96666667 1.         0.86666667 0.96666667]
ExtraTreesClassifier 의 평균  acc:  0.9467
GaussianNB 의 acc [0.96666667 0.9        1.         0.9        0.96666667]
GaussianNB 의 평균  acc:  0.9467
GaussianProcessClassifier 의 acc [0.96666667 0.96666667 1.         0.9        0.96666667]
GaussianProcessClassifier 의 평균  acc:  0.96
GradientBoostingClassifier 의 acc [0.93333333 0.96666667 1.         0.93333333 0.96666667]
GradientBoostingClassifier 의 평균  acc:  0.96
HistGradientBoostingClassifier 의 acc [0.86666667 0.96666667 1.         0.9        0.96666667]
HistGradientBoostingClassifier 의 평균  acc:  0.94
KNeighborsClassifier 의 acc [0.96666667 0.96666667 1.         0.9        0.96666667]
KNeighborsClassifier 의 평균  acc:  0.96
LabelPropagation 의 acc [0.93333333 1.         1.         0.9        0.96666667]
LabelPropagation 의 평균  acc:  0.96
LabelSpreading 의 acc [0.93333333 1.         1.         0.9        0.96666667]
LabelSpreading 의 평균  acc:  0.96
LinearDiscriminantAnalysis 의 acc [1.  1.  1.  0.9 1. ]
LinearDiscriminantAnalysis 의 평균  acc:  0.98
LinearSVC 의 acc [0.96666667 0.96666667 1.         0.9        1.        ]
LinearSVC 의 평균  acc:  0.9667
LogisticRegression 의 acc [1.         0.96666667 1.         0.9        0.96666667]
LogisticRegression 의 평균  acc:  0.9667
LogisticRegressionCV 의 acc [1.         0.96666667 1.         0.9        1.        ]
LogisticRegressionCV 의 평균  acc:  0.9733
MLPClassifier 의 acc [0.96666667 0.96666667 1.         0.93333333 1.        ]
MLPClassifier 의 평균  acc:  0.9733
MultiOutputClassifier 은 없다
MultinomialNB 의 acc [0.96666667 0.93333333 1.         0.93333333 1.        ]
MultinomialNB 의 평균  acc:  0.9667
NearestCentroid 의 acc [0.93333333 0.9        0.96666667 0.9        0.96666667]
NearestCentroid 의 평균  acc:  0.9333
NuSVC 의 acc [0.96666667 0.96666667 1.         0.93333333 1.        ]
NuSVC 의 평균  acc:  0.9733
OneVsOneClassifier 은 없다
OneVsRestClassifier 은 없다
OutputCodeClassifier 은 없다
PassiveAggressiveClassifier 의 acc [0.66666667 0.86666667 0.8        0.86666667 0.86666667]
PassiveAggressiveClassifier 의 평균  acc:  0.8133
Perceptron 의 acc [0.66666667 0.66666667 0.93333333 0.73333333 0.9       ]
Perceptron 의 평균  acc:  0.78
QuadraticDiscriminantAnalysis 의 acc [1.         0.96666667 1.         0.93333333 1.        ]
QuadraticDiscriminantAnalysis 의 평균  acc:  0.98
RadiusNeighborsClassifier 의 acc [0.96666667 0.9        0.96666667 0.93333333 1.        ]
RadiusNeighborsClassifier 의 평균  acc:  0.9533
RandomForestClassifier 의 acc [1.         0.96666667 1.         0.9        0.96666667]
RandomForestClassifier 의 평균  acc:  0.9667
RidgeClassifier 의 acc [0.86666667 0.8        0.93333333 0.7        0.9       ]
RidgeClassifier 의 평균  acc:  0.84
RidgeClassifierCV 의 acc [0.86666667 0.8        0.93333333 0.7        0.9       ]
RidgeClassifierCV 의 평균  acc:  0.84
SGDClassifier 의 acc [0.7        0.63333333 0.93333333 0.93333333 0.7       ]
SGDClassifier 의 평균  acc:  0.78
SVC 의 acc [0.96666667 0.96666667 1.         0.93333333 0.96666667]
SVC 의 평균  acc:  0.9667
StackingClassifier 은 없다
VotingClassifier 은 없다
'''



'''#3. training
model.fit(x_train, y_train)

#4.predict
y_pred = model.predict(x_test)
y_pred = np.rint(y_pred) # ->  반올림을 통해 0 아니면 1의 값으로 y_pred를 변형 
print(x_train,'의 예측결과 : ', y_pred)

result = model.score(x_test, y_test)
print('model.score: ', result)

acc = accuracy_score(y_test, y_pred)
print('accuracy_score: ', acc)'''

'''
loss:  0.10622021555900574
accuracy:  0.9555555582046509

'''