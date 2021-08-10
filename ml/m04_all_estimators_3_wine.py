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
from sklearn.datasets import load_wine
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')

datasets = load_wine()

print(type(datasets))  # <class 'sklearn.utils.Bunch'>
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target

print(x.shape) # (178, 13)
print(y.shape) # (178, )

# y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
test_size=0.7, random_state=9, shuffle=True)

scaler = MinMaxScaler()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)


#2.modeling
allAlgorithms = all_estimators(type_filter='classifier')

print(allAlgorithms)
# [('AdaBoostClassifier', <class 'sklearn.ensemble._weight_boosting.AdaBoostClassifier'>), 이런식으로 출력됨 
print(len(allAlgorithms))
# 모델의 개수는 41개 

for (name, algorithm) in allAlgorithms:
    try:

        model = algorithm()

        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        print(name,'의 정답률 : ', acc)

    except:
        print(name,'은 없다 ')
        continue
    
'''
AdaBoostClassifier 의 정답률 :  0.816
BaggingClassifier 의 정답률 :  0.92
BernoulliNB 의 정답률 :  0.376
CalibratedClassifierCV 의 정답률 :  0.896
CategoricalNB 은 없다 
ClassifierChain 은 없다
ComplementNB 의 정답률 :  0.648
DecisionTreeClassifier 의 정답률 :  0.888
DummyClassifier 의 정답률 :  0.376
ExtraTreeClassifier 의 정답률 :  0.872
ExtraTreesClassifier 의 정답률 :  0.984
GaussianNB 의 정답률 :  0.952
GaussianProcessClassifier 의 정답률 :  0.376
GradientBoostingClassifier 의 정답률 :  0.936
HistGradientBoostingClassifier 의 정답률 :  0.92
KNeighborsClassifier 의 정답률 :  0.656
LabelPropagation 의 정답률 :  0.432
LabelSpreading 의 정답률 :  0.432
LinearDiscriminantAnalysis 의 정답률 :  0.96
LinearSVC 의 정답률 :  0.704
LogisticRegression 의 정답률 :  0.936
LogisticRegressionCV 의 정답률 :  0.896
MLPClassifier 의 정답률 :  0.904
MultiOutputClassifier 은 없다 
MultinomialNB 의 정답률 :  0.872
NearestCentroid 의 정답률 :  0.752
NuSVC 의 정답률 :  0.848
OneVsOneClassifier 은 없다
OneVsRestClassifier 은 없다
OutputCodeClassifier 은 없다
PassiveAggressiveClassifier 의 정답률 :  0.464
Perceptron 의 정답률 :  0.4
QuadraticDiscriminantAnalysis 의 정답률 :  0.504
RadiusNeighborsClassifier 은 없다
RandomForestClassifier 의 정답률 :  0.968
RidgeClassifier 의 정답률 :  0.952
RidgeClassifierCV 의 정답률 :  0.952
SGDClassifier 의 정답률 :  0.632
SVC 의 정답률 :  0.672
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