import numpy as np
from sklearn.datasets import load_breast_cancer
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

datasets = load_breast_cancer()

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
AdaBoostClassifier 의 acc [0.94736842 0.99122807 0.94736842 0.96491228 0.97345133]
AdaBoostClassifier 의 평균  acc:  0.9649
BaggingClassifier 의 acc [0.92982456 0.96491228 0.92982456 0.93859649 0.96460177]
BaggingClassifier 의 평균  acc:  0.9456
BernoulliNB 의 acc [0.64035088 0.65789474 0.62280702 0.5877193  0.62831858]
BernoulliNB 의 평균  acc:  0.6274
CalibratedClassifierCV 의 acc [0.89473684 0.93859649 0.89473684 0.92982456 0.97345133]
CalibratedClassifierCV 의 평균  acc:  0.9263
CategoricalNB 의 acc [nan nan nan nan nan]
CategoricalNB 의 평균  acc:  nan
ClassifierChain 은 없다
ComplementNB 의 acc [0.86842105 0.92982456 0.87719298 0.9122807  0.89380531]
ComplementNB 의 평균  acc:  0.8963
DecisionTreeClassifier 의 acc [0.92105263 0.94736842 0.9122807  0.89473684 0.92920354]
DecisionTreeClassifier 의 평균  acc:  0.9209
DummyClassifier 의 acc [0.64035088 0.65789474 0.62280702 0.5877193  0.62831858]
DummyClassifier 의 평균  acc:  0.6274
ExtraTreeClassifier 의 acc [0.88596491 0.93859649 0.87719298 0.90350877 0.92920354]
ExtraTreeClassifier 의 평균  acc:  0.9069
ExtraTreesClassifier 의 acc [0.96491228 0.98245614 0.96491228 0.95614035 0.99115044]
ExtraTreesClassifier 의 평균  acc:  0.9719
GaussianNB 의 acc [0.93859649 0.96491228 0.9122807  0.93859649 0.95575221]
GaussianNB 의 평균  acc:  0.942
GaussianProcessClassifier 의 acc [0.87719298 0.89473684 0.89473684 0.94736842 0.94690265]
GaussianProcessClassifier 의 평균  acc:  0.9122
GradientBoostingClassifier 의 acc [0.94736842 0.97368421 0.95614035 0.93859649 0.98230088]
GradientBoostingClassifier 의 평균  acc:  0.9596
HistGradientBoostingClassifier 의 acc [0.97368421 0.98245614 0.96491228 0.96491228 0.98230088]
HistGradientBoostingClassifier 의 평균  acc:  0.9737
KNeighborsClassifier 의 acc [0.92105263 0.92105263 0.92105263 0.92105263 0.95575221]
KNeighborsClassifier 의 평균  acc:  0.928
LabelPropagation 의 acc [0.36842105 0.35964912 0.4122807  0.42105263 0.38938053]
LabelPropagation 의 평균  acc:  0.3902
LabelSpreading 의 acc [0.36842105 0.35964912 0.4122807  0.42105263 0.38938053]
LabelSpreading 의 평균  acc:  0.3902
LinearDiscriminantAnalysis 의 acc [0.94736842 0.98245614 0.94736842 0.95614035 0.97345133]
LinearDiscriminantAnalysis 의 평균  acc:  0.9614
LinearSVC 의 acc [0.87719298 0.93859649 0.89473684 0.87719298 0.85840708]
LinearSVC 의 평균  acc:  0.8892
LogisticRegression 의 acc [0.93859649 0.95614035 0.88596491 0.94736842 0.96460177]
LogisticRegression 의 평균  acc:  0.9385
LogisticRegressionCV 의 acc [0.95614035 0.97368421 0.90350877 0.96491228 0.96460177]
LogisticRegressionCV 의 평균  acc:  0.9526
MLPClassifier 의 acc [0.90350877 0.94736842 0.90350877 0.92982456 0.97345133]
MLPClassifier 의 평균  acc:  0.9315
MultiOutputClassifier 은 없다
MultinomialNB 의 acc [0.85964912 0.92105263 0.87719298 0.9122807  0.89380531]
MultinomialNB 의 평균  acc:  0.8928
NearestCentroid 의 acc [0.86842105 0.89473684 0.85964912 0.9122807  0.91150442]
NearestCentroid 의 평균  acc:  0.8893
NuSVC 의 acc [0.85964912 0.9122807  0.83333333 0.87719298 0.88495575]
NuSVC 의 평균  acc:  0.8735
OneVsOneClassifier 은 없다
OneVsRestClassifier 은 없다
OutputCodeClassifier 은 없다
PassiveAggressiveClassifier 의 acc [0.89473684 0.92982456 0.86842105 0.8245614  0.96460177]
PassiveAggressiveClassifier 의 평균  acc:  0.8964
Perceptron 의 acc [0.40350877 0.80701754 0.85964912 0.86842105 0.94690265]
Perceptron 의 평균  acc:  0.7771
QuadraticDiscriminantAnalysis 의 acc [0.93859649 0.95614035 0.93859649 0.98245614 0.94690265]
QuadraticDiscriminantAnalysis 의 평균  acc:  0.9525
RadiusNeighborsClassifier 의 acc [nan nan nan nan nan]
RadiusNeighborsClassifier 의 평균  acc:  nan
RandomForestClassifier 의 acc [0.97368421 0.96491228 0.96491228 0.93859649 0.98230088]
RandomForestClassifier 의 평균  acc:  0.9649
RidgeClassifier 의 acc [0.95614035 0.98245614 0.92105263 0.95614035 0.95575221]
RidgeClassifier 의 평균  acc:  0.9543
RidgeClassifierCV 의 acc [0.94736842 0.97368421 0.93859649 0.95614035 0.96460177]
RidgeClassifierCV 의 평균  acc:  0.9561
SGDClassifier 의 acc [0.87719298 0.65789474 0.88596491 0.83333333 0.77876106]
SGDClassifier 의 평균  acc:  0.8066
SVC 의 acc [0.89473684 0.92982456 0.89473684 0.92105263 0.96460177]
SVC 의 평균  acc:  0.921
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