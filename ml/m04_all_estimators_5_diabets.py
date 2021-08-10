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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.datasets import load_diabetes
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')


datasets  = load_diabetes()
x = datasets.data
y = datasets.target

print(datasets.DESCR)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
train_size=0.8, shuffle=True, random_state=9)

scaler =  PowerTransformer()
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
ARDRegression 의 정답률 :  0.5894059579985
AdaBoostRegressor 의 정답률 :  0.5070664643498386
BaggingRegressor 의 정답률 :  0.534628718419449
BayesianRidge 의 정답률 :  0.5964849213159903
CCA 의 정답률 :  0.5852713803269576
DecisionTreeRegressor 의 정답률 :  -0.25886019453215225
DummyRegressor 의 정답률 :  -0.01545589029660177
ElasticNet 의 정답률 :  -0.005148552241826643
ElasticNetCV 의 정답률 :  0.5304646671601247
ExtraTreeRegressor 의 정답률 :  0.05116853890148909
ExtraTreesRegressor 의 정답률 :  0.5528009114672563
GammaRegressor 의 정답률 :  -0.007001976289833234
GaussianProcessRegressor 의 정답률 :  -15.15467471601707
GradientBoostingRegressor 의 정답률 :  0.5469827115277309
HistGradientBoostingRegressor 의 정답률 :  0.5390878373341259
HuberRegressor 의 정답률 :  0.5825758702996858
IsotonicRegression 은 없다
KNeighborsRegressor 의 정답률 :  0.48322394424037063
KernelRidge 의 정답률 :  -3.4724121351281463
Lars 의 정답률 :  0.5851141269959733
LarsCV 의 정답률 :  0.5896206660679899
Lasso 의 정답률 :  0.4141675244086438
LassoCV 의 정답률 :  0.5910251294317925
LassoLars 의 정답률 :  0.45238726393914497
LassoLarsCV 의 정답률 :  0.5896206660679899
LassoLarsIC 의 정답률 :  0.5962837270477235  << best!
LinearRegression 의 정답률 :  0.5851141269959736
LinearSVR 의 정답률 :  -0.2797387315384048
MLPRegressor 의 정답률 :  -3.0592636163323945
MultiOutputRegressor 은 없다
MultiTaskElasticNet 은 없다
MultiTaskElasticNetCV 은 없다
MultiTaskLasso 은 없다
MultiTaskLassoCV 은 없다
NuSVR 의 정답률 :  0.17896352175589791
OrthogonalMatchingPursuit 의 정답률 :  0.32241768669099435
OrthogonalMatchingPursuitCV 의 정답률 :  0.5812114430406672
PLSCanonical 의 정답률 :  -1.6878518911601064
PLSRegression 의 정답률 :  0.6072433305368463
PassiveAggressiveRegressor 의 정답률 :  0.5540620295981443
PoissonRegressor 의 정답률 :  0.4011652404741377
RANSACRegressor 의 정답률 :  0.34726413006226664
RadiusNeighborsRegressor 의 정답률 :  -0.01545589029660177
RandomForestRegressor 의 정답률 :  0.540687989458144
RegressorChain 은 없다
Ridge 의 정답률 :  0.504253823751757
RidgeCV 의 정답률 :  0.5943421168074103
SGDRegressor 의 정답률 :  0.48549631506523794
SVR 의 정답률 :  0.21089225663642996
StackingRegressor 은 없다
TheilSenRegressor 의 정답률 :  0.5917876871840357
TransformedTargetRegressor 의 정답률 :  0.5851141269959736
TweedieRegressor 의 정답률 :  -0.0077214185844955985
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