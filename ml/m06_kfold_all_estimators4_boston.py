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

#2.modeling
allAlgorithms = all_estimators(type_filter='regressor')

print(allAlgorithms)
# [('AdaBoostClassifier', <class 'sklearn.ensemble._weight_boosting.AdaBoostClassifier'>), 이런식으로 출력됨 
print(len(allAlgorithms))
# 모델의 개수는 54 

for (name, algorithm) in allAlgorithms:
    try:

        model = algorithm()

        model.fit(x, y)

        y_pred = model.predict(x)


        from sklearn.metrics import r2_score
        r2 = r2_score(y, y_pred) # y_test, y_pred 차이

        print(name,'의 r2 스코어 : ', r2)

    except:
        print(name,'은 없다 ')
        continue
    
'''
ARDRegression 의 r2 스코어 :  0.7339007153830579
AdaBoostRegressor 의 r2 스코어 :  0.8986677152275655
BaggingRegressor 의 r2 스코어 :  0.976283825875879
BayesianRidge 의 r2 스코어 :  0.7312094189729963
CCA 의 r2 스코어 :  0.6938661491345115
DecisionTreeRegressor 의 r2 스코어 :  1.0
DummyRegressor 의 r2 스코어 :  0.0
ElasticNet 의 r2 스코어 :  0.6861018474345026
ElasticNetCV 의 r2 스코어 :  0.6712548925406525
ExtraTreeRegressor 의 r2 스코어 :  1.0
ExtraTreesRegressor 의 r2 스코어 :  1.0
GammaRegressor 의 r2 스코어 :  2.220446049250313e-16
GaussianProcessRegressor 의 r2 스코어 :  1.0
GradientBoostingRegressor 의 r2 스코어 :  0.9761405838418584
HistGradientBoostingRegressor 의 r2 스코어 :  0.9812280301907457
HuberRegressor 의 r2 스코어 :  0.658658112409731
IsotonicRegression 은 없다
KNeighborsRegressor 의 r2 스코어 :  0.716098217736928
KernelRidge 의 r2 스코어 :  0.7136796967433584
Lars 의 r2 스코어 :  0.7405955522829124
LarsCV 의 r2 스코어 :  0.7271664273993774
Lasso 의 r2 스코어 :  0.6825842212709925
LassoCV 의 r2 스코어 :  0.7024437179872696
LassoLars 의 r2 스코어 :  0.0
LassoLarsCV 의 r2 스코어 :  0.7293509716983886
LassoLarsIC 의 r2 스코어 :  0.7404629491704635
LinearRegression 의 r2 스코어 :  0.7406426641094095
LinearSVR 의 r2 스코어 :  0.40410042544122315
MLPRegressor 의 r2 스코어 :  0.6751819789263717
MultiOutputRegressor 은 없다 
MultiTaskElasticNet 은 없다
MultiTaskElasticNetCV 은 없다
MultiTaskLasso 은 없다
MultiTaskLassoCV 은 없다
NuSVR 의 r2 스코어 :  0.24083334537437706
OrthogonalMatchingPursuit 의 r2 스코어 :  0.5441462975864797
OrthogonalMatchingPursuitCV 의 r2 스코어 :  0.6786241601613112
PLSCanonical 의 r2 스코어 :  -2.113310285990865
PLSRegression 의 r2 스코어 :  0.7063745792855106
PassiveAggressiveRegressor 의 r2 스코어 :  -0.03311277905595622
PoissonRegressor 의 r2 스코어 :  0.7826114485428501
RANSACRegressor 의 r2 스코어 :  0.5583065271188734
RadiusNeighborsRegressor 의 r2 스코어 :  0.9990935543538173
RandomForestRegressor 의 r2 스코어 :  0.9821360104244506
RegressorChain 은 없다 
Ridge 의 r2 스코어 :  0.7388703133867616
RidgeCV 의 r2 스코어 :  0.740600292222802
SGDRegressor 의 r2 스코어 :  -2.252441088225952e+25
SVR 의 r2 스코어 :  0.20849811543173336
StackingRegressor 은 없다 
TheilSenRegressor 의 r2 스코어 :  0.6992023334562603
TransformedTargetRegressor 의 r2 스코어 :  0.7406426641094095
TweedieRegressor 의 r2 스코어 :  0.6679465649687071
VotingRegressor 은 없다 
'''
