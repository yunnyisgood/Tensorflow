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
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, StandardScaler, MinMaxScaler
from sklearn.datasets import load_boston 
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')


#1. data
datasets = load_boston()
x = datasets.data
y = datasets.target

print(x.shape) # (506, 13) -> 13가지의 특성 -> Input_dim = 13
print(y.shape) # (506,) -> output_dim = 1

print(datasets.feature_names)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
train_size=0.7, shuffle=True, random_state=9)

scaler = PowerTransformer()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)


#2.modeling
allAlgorithms = all_estimators(type_filter='regressor')

print(allAlgorithms)
# [('AdaBoostClassifier', <class 'sklearn.ensemble._weight_boosting.AdaBoostClassifier'>), 이런식으로 출력됨 
print(len(allAlgorithms))
# 모델의 개수는 54 

for (name, algorithm) in allAlgorithms:
    try:

        model = algorithm()

        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)

        result = model.score(x_test, y_test)
        print(name,'의 정답률 : ', result)

    except:
        print(name,'은 없다 ')
        continue
    
'''
ARDRegression 의 정답률 :  0.7597725309740284
AdaBoostRegressor 의 정답률 :  0.8288638472683255
BaggingRegressor 의 정답률 :  0.8558842036884855
BayesianRidge 의 정답률 :  0.7626822452838828
CCA 의 정답률 :  0.7608263146559946
DecisionTreeRegressor 의 정답률 :  0.6303881826655491
DummyRegressor 의 정답률 :  -0.019031831720200065
ElasticNet 의 정답률 :  0.7041261455492027
ElasticNetCV 의 정답률 :  0.688208425363434
ExtraTreeRegressor 의 정답률 :  0.7617565686338532
ExtraTreesRegressor 의 정답률 :  0.8757220158725292
GammaRegressor 의 정답률 :  -0.021431286213422496
GaussianProcessRegressor 의 정답률 :  -6.1502314762299966
GradientBoostingRegressor 의 정답률 :  0.8843714153543896 << best!
HistGradientBoostingRegressor 의 정답률 :  0.859634619105514
HuberRegressor 의 정답률 :  0.6571530919435165
IsotonicRegression 은 없다
KNeighborsRegressor 의 정답률 :  0.5941848521354716
KernelRidge 의 정답률 :  0.7462027488385179
Lars 의 정답률 :  0.7826126074271014
LarsCV 의 정답률 :  0.7826126074271014
Lasso 의 정답률 :  0.693670187075003
LassoCV 의 정답률 :  0.7253711318045533
LassoLars 의 정답률 :  -0.019031831720200065
LassoLarsCV 의 정답률 :  0.7826126074271014
LassoLarsIC 의 정답률 :  0.7788514319949902
LinearRegression 의 정답률 :  0.7826126074271011
LinearSVR 의 정답률 :  0.5537668191630225
MLPRegressor 의 정답률 :  0.6020282062339251
MultiOutputRegressor 은 없다 
MultiTaskElasticNet 은 없다
MultiTaskElasticNetCV 은 없다
MultiTaskLasso 은 없다
MultiTaskLassoCV 은 없다
NuSVR 의 정답률 :  0.21638315186027735
OrthogonalMatchingPursuit 의 정답률 :  0.5545805166503027
OrthogonalMatchingPursuitCV 의 정답률 :  0.7310024148963548
PLSCanonical 의 정답률 :  -1.593792255202167
PLSRegression 의 정답률 :  0.746876771020851
PassiveAggressiveRegressor 의 정답률 :  0.14147232339762594
PoissonRegressor 의 정답률 :  0.8053503463300503
RANSACRegressor 의 정답률 :  0.5424241579202363
RadiusNeighborsRegressor 은 없다 
RandomForestRegressor 의 정답률 :  0.8361721038751012
RegressorChain 은 없다 
Ridge 의 정답률 :  0.7766313587327416
RidgeCV 의 정답률 :  0.781888419918507
SGDRegressor 의 정답률 :  -3.21072490861415e+25
SVR 의 정답률 :  0.18765807859708883
StackingRegressor 은 없다
TheilSenRegressor 의 정답률 :  0.7286121661069412
TransformedTargetRegressor 의 정답률 :  0.7826126074271011
TweedieRegressor 의 정답률 :  0.6823492890873477
VotingRegressor 은 없다
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