import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Reshape, Conv1D
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, QuantileTransformer, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline, Pipeline
import time
import warnings
warnings.filterwarnings('ignore')


'''
m31로 만든 0.95이상의 n_components를 사용하여
xgb 모델을 만들 것 (디폴트)

mnist dnn보다 성능 좋게 만들기
dnn, cnn 비교

RandomSearch로도 해볼것

'''
(x_train, y_train), (x_test, y_test) = mnist.load_data() 
# 

print(x_train.shape, x_test.shape)
# (60000, 28, 28) (10000, 28, 28)

x = np.append(x_train, x_test, axis=0)
print(x.shape) # (70000, 28, 28)
print(type(x))  # <class 'numpy.ndarray'>   

x = x.reshape(70000, 28*28)

pca = PCA(n_components=154)
x = pca.fit_transform(x)
pca_EVR = pca.explained_variance_ratio_
print(pca_EVR)
print(sum(pca_EVR))

cumsum = np.cumsum(pca_EVR)
print(cumsum)
print(np.argmax(cumsum >= 0.95)+1)


x_train, x_test, y_train, y_test = train_test_split(
    x, np.concatenate((y_train, y_test)), train_size=0.8, random_state=66, shuffle=True
)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (56000, 154) (14000, 154) (56000,) (14000,)

#교차검증
n_splits = 5
kfold = KFold(n_splits=n_splits,  shuffle=True, random_state=66)


parameters = [

    {"xgb__n_estimators":[100, 200, 300], "xgb__learning_rate": [0.1, 0.3, 0.001, 0.01],
    "xgb__max_depth": [4,5,6]},
    {"xgb__n_estimators":[90, 100, 110], "xgb__learning_rate": [0.1,  0.001, 0.01],
    "xgb__max_depth": [4,5,6], "xgb__colsample_bytree":[0.6, 0.9, 1]},
     {"xgb__n_estimators":[90, 110], "xgb__learning_rate": [0.1,  0.001, 0.5],
    "xgb__max_depth": [4,5,6], "xgb__colsample_bytree":[0.6, 0.9, 1],
    "xgb__colsample_bylevel": [0.6, 0.7, 0.9]}

]
n_jobs = -1

pipe = Pipeline([("scaler", MinMaxScaler()), ("xgb", XGBClassifier())])
# alias를 사용할 수 있다 

model = RandomizedSearchCV(pipe, parameters, cv=kfold, verbose=1)

#3. training
start_time = time.time()
model.fit(x_train, y_train)

#4.predict
# 4-1 : train값으로 훈련을 했을 때 정확도
print("최적의 매개변수: ", model.best_estimator_)
print("best_params: ", model.best_params_)
print("best_score_: ", model.best_score_)

# 4-2 : test값을 따로 빼서 훈련을 거치지 않은 값들로 학습을 시킨 뒤 평가했을 때 
print("model.score: ", model.score(x_test, y_test))

y_pred  = model.predict(x_test)
print("정답률: ", accuracy_score(y_test, y_pred))


'''
압축 전 DNN:
loss:  2.3022656440734863
accuracy:  0.10971428453922272

압축 후 DNN:
loss:  0.34136977791786194
accuracy:  0.946142852306366

XGBClassifier, RandomizedSearchCV PCA(n_components=154)
최적의 매개변수:  Pipeline(steps=[('scaler', MinMaxScaler()),
                ('xgb',
                 XGBClassifier(base_score=0.5, booster='gbtree',
                               colsample_bylevel=1, colsample_bynode=1,
                               colsample_bytree=0.6, gamma=0, gpu_id=-1,
                               importance_type='gain',
                               interaction_constraints='', learning_rate=0.1,
                               max_delta_step=0, max_depth=6,
                               min_child_weight=1, missing=nan,
                               monotone_constraints='()', n_estimators=90,
                               n_jobs=8, num_parallel_tree=1,
                               objective='multi:softprob', random_state=0,
                               reg_alpha=0, reg_lambda=1, scale_pos_weight=None,
                               subsample=1, tree_method='exact',
                               validate_parameters=1, verbosity=None))])
best_params:  {'xgb__n_estimators': 90, 'xgb__max_depth': 6, 'xgb__learning_rate': 0.1, 'xgb__colsample_bytree': 0.6}
best_score_:  0.9484464285714287
model.score:  0.9450714285714286
정답률:  0.9450714285714286

'''