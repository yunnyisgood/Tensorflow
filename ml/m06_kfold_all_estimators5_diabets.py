import numpy as np
from sklearn.datasets import load_diabetes
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
datasets = load_diabetes()
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
ARDRegression 의 r2 스코어 :  0.5132304371961408
AdaBoostRegressor 의 r2 스코어 :  0.6163595739625143
BaggingRegressor 의 r2 스코어 :  0.8938512394305232
BayesianRidge 의 r2 스코어 :  0.5150901337586229
CCA 의 r2 스코어 :  0.5018962478282587
DecisionTreeRegressor 의 r2 스코어 :  1.0
DummyRegressor 의 r2 스코어 :  0.0
ElasticNet 의 r2 스코어 :  0.008834770038328776
ElasticNetCV 의 r2 스코어 :  0.4544662491541832
ExtraTreeRegressor 의 r2 스코어 :  1.0
ExtraTreesRegressor 의 r2 스코어 :  1.0
GammaRegressor 의 r2 스코어 :  0.006517548628182213
GaussianProcessRegressor 의 r2 스코어 :  0.9591627077060221
GradientBoostingRegressor 의 r2 스코어 :  0.7990392018966864
HistGradientBoostingRegressor 의 r2 스코어 :  0.9299589575098558
HuberRegressor 의 r2 스코어 :  0.5149680646520973
IsotonicRegression 은 없다
KNeighborsRegressor 의 r2 스코어 :  0.604957605699507
KernelRidge 의 r2 스코어 :  -3.451811818789741
Lars 의 r2 스코어 :  0.5177494254132934
LarsCV 의 r2 스코어 :  0.5055335801649771
Lasso 의 r2 스코어 :  0.35737932948734685
LassoCV 의 r2 스코어 :  0.517422065602023
LassoLars 의 r2 스코어 :  0.35738113086922063
LassoLarsCV 의 r2 스코어 :  0.5173975480346735
LassoLarsIC 의 r2 스코어 :  0.5134108568681748
LinearRegression 의 r2 스코어 :  0.5177494254132934
LinearSVR 의 r2 스코어 :  -0.28161368795996977
MLPRegressor 의 r2 스코어 :  -1.9117604772323382
MultiOutputRegressor 은 없다 
MultiTaskElasticNet 은 없다
MultiTaskElasticNetCV 은 없다
MultiTaskLasso 은 없다
MultiTaskLassoCV 은 없다
NuSVR 의 r2 스코어 :  0.20366477711531272
OrthogonalMatchingPursuit 의 r2 스코어 :  0.3439237602253803
OrthogonalMatchingPursuitCV 의 r2 스코어 :  0.5086324897618718
PLSCanonical 의 r2 스코어 :  -1.1744670155732195
PLSRegression 의 r2 스코어 :  0.5083254817781028
PassiveAggressiveRegressor 의 r2 스코어 :  0.49348985503862475
PoissonRegressor 의 r2 스코어 :  0.3531664466546165
RANSACRegressor 의 r2 스코어 :  0.27551477583413364
RadiusNeighborsRegressor 의 r2 스코어 :  0.0
RandomForestRegressor 의 r2 스코어 :  0.9206580564862106
RegressorChain 은 없다 
Ridge 의 r2 스코어 :  0.4512313946799056
RidgeCV 의 r2 스코어 :  0.5125629767961007
SGDRegressor 의 r2 스코어 :  0.43676248830441866
SVR 의 r2 스코어 :  0.2071794500005485
StackingRegressor 은 없다
TheilSenRegressor 의 r2 스코어 :  0.5134614490533168
TransformedTargetRegressor 의 r2 스코어 :  0.5177494254132934
TweedieRegressor 의 r2 스코어 :  0.0065218716081014705
VotingRegressor 은 없다
'''
