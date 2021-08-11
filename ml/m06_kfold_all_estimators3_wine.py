import numpy as np
from sklearn.datasets import load_wine
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

datasets = load_wine()

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
AdaBoostClassifier 의 acc [0.88888889 0.86111111 0.88888889 0.94285714 0.97142857]
AdaBoostClassifier 의 평균  acc:  0.9106
BaggingClassifier 의 acc [1.         0.91666667 0.91666667 0.97142857 1.        ]
BaggingClassifier 의 평균  acc:  0.961
BernoulliNB 의 acc [0.41666667 0.47222222 0.27777778 0.48571429 0.34285714]
BernoulliNB 의 평균  acc:  0.399
CalibratedClassifierCV 의 acc [0.94444444 0.94444444 0.88888889 0.88571429 0.91428571]
CalibratedClassifierCV 의 평균  acc:  0.9156
CategoricalNB 의 acc [       nan        nan        nan 0.94285714        nan]
CategoricalNB 의 평균  acc:  nan
ClassifierChain 은 없다
ComplementNB 의 acc [0.69444444 0.80555556 0.55555556 0.6        0.6       ]
ComplementNB 의 평균  acc:  0.6511
DecisionTreeClassifier 의 acc [0.94444444 0.97222222 0.91666667 0.85714286 0.88571429]
DecisionTreeClassifier 의 평균  acc:  0.9152
DummyClassifier 의 acc [0.41666667 0.47222222 0.27777778 0.48571429 0.34285714]
DummyClassifier 의 평균  acc:  0.399
ExtraTreeClassifier 의 acc [0.91666667 0.86111111 0.86111111 0.88571429 0.91428571]
ExtraTreeClassifier 의 평균  acc:  0.8878
ExtraTreesClassifier 의 acc [1.         0.97222222 1.         0.97142857 1.        ]
ExtraTreesClassifier 의 평균  acc:  0.9887
GaussianNB 의 acc [1.         0.91666667 0.97222222 0.97142857 1.        ]
GaussianNB 의 평균  acc:  0.9721
GaussianProcessClassifier 의 acc [0.44444444 0.30555556 0.55555556 0.62857143 0.45714286]
GaussianProcessClassifier 의 평균  acc:  0.4783
GradientBoostingClassifier 의 acc [0.97222222 0.91666667 0.88888889 0.97142857 0.97142857]
GradientBoostingClassifier 의 평균  acc:  0.9441
HistGradientBoostingClassifier 의 acc [0.97222222 0.94444444 1.         0.97142857 1.        ]
HistGradientBoostingClassifier 의 평균  acc:  0.9776
KNeighborsClassifier 의 acc [0.69444444 0.77777778 0.61111111 0.62857143 0.74285714]
KNeighborsClassifier 의 평균  acc:  0.691
LabelPropagation 의 acc [0.52777778 0.47222222 0.5        0.4        0.54285714]
LabelPropagation 의 평균  acc:  0.4886
LabelSpreading 의 acc [0.52777778 0.47222222 0.5        0.4        0.54285714]
LabelSpreading 의 평균  acc:  0.4886
LinearDiscriminantAnalysis 의 acc [1.         0.97222222 1.         0.97142857 1.        ]
LinearDiscriminantAnalysis 의 평균  acc:  0.9887
LinearSVC 의 acc [0.97222222 0.88888889 0.86111111 0.71428571 0.91428571]
LinearSVC 의 평균  acc:  0.8702
LogisticRegression 의 acc [0.97222222 0.94444444 0.94444444 0.94285714 1.        ]
LogisticRegression 의 평균  acc:  0.9608
LogisticRegressionCV 의 acc [0.97222222 0.91666667 0.97222222 0.94285714 0.97142857]
LogisticRegressionCV 의 평균  acc:  0.9551
MLPClassifier 의 acc [0.5        0.91666667 0.5        0.88571429 0.88571429]
MLPClassifier 의 평균  acc:  0.7376
MultiOutputClassifier 은 없다
MultinomialNB 의 acc [0.77777778 0.91666667 0.86111111 0.82857143 0.82857143]
MultinomialNB 의 평균  acc:  0.8425
NearestCentroid 의 acc [0.69444444 0.72222222 0.69444444 0.77142857 0.74285714]
NearestCentroid 의 평균  acc:  0.7251
NuSVC 의 acc [0.91666667 0.86111111 0.91666667 0.85714286 0.8       ]
NuSVC 의 평균  acc:  0.8703
OneVsOneClassifier 은 없다
OneVsRestClassifier 은 없다
OutputCodeClassifier 은 없다
PassiveAggressiveClassifier 의 acc [0.36111111 0.72222222 0.52777778 0.62857143 0.6       ]
PassiveAggressiveClassifier 의 평균  acc:  0.5679
Perceptron 의 acc [0.61111111 0.80555556 0.47222222 0.48571429 0.62857143]
Perceptron 의 평균  acc:  0.6006
QuadraticDiscriminantAnalysis 의 acc [0.97222222 1.         1.         1.         1.        ]
QuadraticDiscriminantAnalysis 의 평균  acc:  0.9944
RadiusNeighborsClassifier 의 acc [nan nan nan nan nan]
RadiusNeighborsClassifier 의 평균  acc:  nan
RandomForestClassifier 의 acc [1.         0.94444444 1.         0.97142857 1.        ]
RandomForestClassifier 의 평균  acc:  0.9832
RidgeClassifier 의 acc [1.         1.         1.         0.97142857 1.        ]
RidgeClassifier 의 평균  acc:  0.9943
RidgeClassifierCV 의 acc [1.         1.         1.         0.97142857 1.        ]
RidgeClassifierCV 의 평균  acc:  0.9943
SGDClassifier 의 acc [0.52777778 0.77777778 0.44444444 0.65714286 0.54285714]
SGDClassifier 의 평균  acc:  0.59
SVC 의 acc [0.69444444 0.69444444 0.61111111 0.62857143 0.6       ]
SVC 의 평균  acc:  0.6457
StackingClassifier 은 없다
VotingClassifier 은 없다
'''



'''
loss:  0.10622021555900574
accuracy:  0.9555555582046509

'''