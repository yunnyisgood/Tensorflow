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
import warnings
warnings.filterwarnings('ignore')


# 다중분류, One-Hot-Encoding 

datasets = load_iris()


x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=9, shuffle=True)


# scaler = MinMaxScaler()
# scaler.fit(x_train)
# scaler.transform(x_train)
# scaler.transform(x_test)

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
AdaBoostClassifier 의 정답률 :  1.0
BaggingClassifier 의 정답률 :  1.0
BernoulliNB 의 정답률 :  0.28888888888888886
CalibratedClassifierCV 의 정답률 :  1.0
CategoricalNB 의 정답률 :  0.9777777777777777
ComplementNB 의 정답률 :  0.6888888888888889
DecisionTreeClassifier 의 정답률 :  1.0
DummyClassifier 의 정답률 :  0.28888888888888886
ExtraTreeClassifier 의 정답률 :  0.9777777777777777      
ExtraTreesClassifier 의 정답률 :  1.0
GaussianNB 의 정답률 :  1.0
GaussianProcessClassifier 의 정답률 :  0.9777777777777777GradientBoostingClassifier 의 정답률 :  1.0
HistGradientBoostingClassifier 의 정답률 :  1.0
KNeighborsClassifier 의 정답률 :  1.0
LabelPropagation 의 정답률 :  1.0
LabelSpreading 의 정답률 :  1.0
LinearDiscriminantAnalysis 의 정답률 :  1.0
LinearSVC 의 정답률 :  1.0
LogisticRegression 의 정답률 :  1.0
LogisticRegressionCV 의 정답률 :  1.0
MLPClassifier 의 정답률 :  1.0
MultinomialNB 의 정답률 :  1.0
NearestCentroid 의 정답률 :  0.9777777777777777
NuSVC 의 정답률 :  0.9777777777777777
PassiveAggressiveClassifier 의 정답률 :  0.8222222222222222
Perceptron 의 정답률 :  0.9111111111111111
QuadraticDiscriminantAnalysis 의 정답률 :  1.0
RadiusNeighborsClassifier 의 정답률 :  0.9777777777777777RandomForestClassifier 의 정답률 :  1.0
RidgeClassifier 의 정답률 :  0.9555555555555556
RidgeClassifierCV 의 정답률 :  0.9555555555555556        
SGDClassifier 의 정답률 :  0.8444444444444444
SVC 의 정답률 :  0.9777777777777777
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