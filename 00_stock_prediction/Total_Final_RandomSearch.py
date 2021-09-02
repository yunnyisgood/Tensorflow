from datetime import timedelta
import pandas as pd
import numpy as np
import yfinance as yf
import tensorflow as tf
from pandas_datareader import data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, QuantileTransformer, StandardScaler
from keras.layers import Dense, LSTM, Input, concatenate, Conv2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
import time
import matplotlib.pyplot as plt
from konlpy.tag import Okt
from icecream import ic
from sklearn.metrics import mean_squared_error
from tensorflow.keras.optimizers import Adam
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import r2_score


'''samsung = pd.read_excel('삼성_6개월_최종2021-08-22_15시29분.xlsx', header=0)
stopwords = pd.read_csv('stopwords.txt').values.tolist()

okt = Okt()

# 다시 형태소 분석
samsung_temp_list = []
for sentence in samsung['title']:
    temp_samsung = []
    temp_samsung = okt.normalize(sentence)
    temp_samsung  = okt.morphs(sentence) #  문장에서 명사 추출 
    temp_samsung = [word for word in temp_samsung if not word in stopwords]  # 불용어 처리 
    samsung_temp_list.append(temp_samsung)
samsung['samsung_temp_list'] = samsung_temp_list

temp_list = []
for i in samsung['word_list']:
    temp = eval(''.join(i))
    print(temp)
    temp_list.append(temp)
print(temp_list)
samsung['score'] = np.nan
samsung['score'] = pd.Series(temp_list)

# 각 기사별 헤드라인의 긍정/부정 최종수치 구하기 

total_score = []
for i in range(len(samsung['samsung_temp_list'])):
    temp_len = len(samsung['samsung_temp_list'][i])
    result = sum(samsung['score'][i])
    avg = result/temp_len # 각 헤드라인별 평균수치
    total_score.append(avg)

samsung['total_score'] = np.nan
samsung['total_score'] = pd.Series(total_score)


# 날짜별 데이터로 합산해서 평균 구하기
grouped = samsung.groupby('date')
df = pd.DataFrame(grouped.mean())


# 주말 반복문으로 다 제거하기 
week_days = pd.date_range(df.index[0], df.index[-1], freq='B')
print('week_days: ', week_days)
for i in df.index:
    if i not in week_days:
        print('index check: ',i)
        df = df.drop([i])
df =df.drop(['2021.02.11.', '2021.03.01.', '2021.05.05.', '2021.05.19.'])

ic('df: ', df)

samsung_score = df['total_score'].to_numpy()

np.save('samsung_save.npy', arr=samsung_score)'''

score = np.load('samsung_save.npy')


# 삼성주가 다운로드
start_date = '2021-02-01'
end_date = '2021-08-01'
SAMSUNG = data.get_data_yahoo('005930.KS', start_date, end_date)
ic(type(SAMSUNG)) #  type(SAMSUNG): <class 'pandas.core.frame.DataFrame'>
ic(SAMSUNG.shape) # SAMSUNG.shape: (125, 6)
ic(SAMSUNG)


dic = {
    'ds': SAMSUNG.index,
    'y': SAMSUNG['Close'],
    'score':score
}

df = pd.DataFrame.from_dict(dic, orient='index')
df = df.transpose()


# x, y 분류
x = df[['y', 'score']]#[:-3]
y = df[['y']]#[3:]

# 판다스 -> 넘파이로 변환 
x = x.to_numpy()
y = y.to_numpy()

def split_x(dataset, size):
    aaa =[]
    for i in range(len(dataset)-size+1): # range(10-4=6) -> 6번동안 반복. 10개의 데이터를 5개씩 분리하기 위한 방법 
        subset = dataset[i : (i+size)] # dataset[0:5] -> dataset 0부터 4번째 값까지 
        aaa.append(subset)
    return np.array(aaa)

size = 5

x = split_x(x, size)
y = split_x(y, size)
print(x.shape, y.shape)
# (121, 5, 2) (121, 5, 1)

x = x.reshape(121*5, 2).astype(float)
y = y.reshape(121*5, 1).astype(float)

x_train, x_test, y_train, y_test = train_test_split(x,  y, 
                                                        train_size=0.7, random_state=9)

ic(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# y_train = y_train.reshape(423, )
# y_test = y_test.reshape(182, )

ic(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# scaling
scaler = MinMaxScaler()
scaler.fit_transform(x_train)
scaler.transform(x_test)

# modeling
n_splits = 5
kfold = KFold(n_splits=n_splits,  shuffle=True, random_state=66)

parameters = [
    {'min_samples_leaf':[3, 5, 7], 'max_depth':[2, 3, 5, 10]},
    { 'min_samples_split':[6, 8, 10]}
]

#2.modeling
model = RandomizedSearchCV(RandomForestRegressor(), parameters, cv=kfold, verbose=1)

es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')

start_time = time.time()
hist = model.fit(x_train, y_train)
end_time = time.time() - start_time

#4.predict
# 4-1 : train값으로 훈련을 했을 때 정확도
print("최적의 매개변수: ", model.best_estimator_)
print("best_params: ", model.best_params_)
print("best_score_: ", model.best_score_)
print("model.score: ", model.score(x_test, y_test))

y_pred  = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
print("r2 스코어: ", r2)

rmse = mean_squared_error(y_test, y_pred)**0.5
print('rmse: ', rmse)


# 시각화 
plt.figure(figsize=(9,5))

# 1
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.plot([y.min(), y.max()], [y.min(), y.max()])
ax.set_xlabel('Actual(y_test)')
ax.set_ylabel('Predicted(y_pred)')
ax.set_title('R2: '+ str(r2))

plt.show()

'''
batch_size = 1
loss:  0.0448458306491375
mse:  0.0448458306491375
r2스코어: 0.9999999840520231
rmse:  0.21176834324036842

batch_size = 8
loss:  0.0017199907451868057
mse:  0.0017199907451868057
r2스코어: 0.9999999993986595
rmse:  0.04112145426689808

batch_size=10
loss:  0.0009225313551723957
mse:  0.0009225313551723957
r2스코어: 0.9999999996733547
rmse:  0.030307244558783326

batch_size=16
loss:  0.001222704304382205
mse:  0.001222704304382205
r2스코어: 0.9999999995651846
rmse:  0.03496718858678711

batch_size=32
loss:  0.005391885992139578
r2스코어: 0.9999999979291958
rmse:  0.07348881040077379

batch_size=64
loss:  0.0009493984980508685
r2스코어: 0.9999999996305625
rmse:  0.031040031216904868

lr = 0.001, batch_Size=10
걸린시간:  2.873347520828247
loss:  0.0005042904522269964
mse:  0.0005042904522269964
r2스코어: 0.9999999998206653
rmse:  0.022456412699081645

batch_size=32
loss:  0.0028886639047414064
r2스코어: 0.9999999989727396
rmse:  0.05374629168965376

batch_size=10, lr=0.1
loss:  561.5869750976562
r2스코어: 0.9998002896588515
rmse:  23.69782652695581

batch_size=10, lr=0.01
걸린시간:  2.4783380031585693
loss:  0.00421943049877882
r2스코어: 0.9999999984621339
rmse:  0.06576086151170994

batch_size=10, lr=0.001
loss:  0.003255875315517187
r2스코어: 0.9999999988667044
rmse:  0.056452064774599464

batch_size=10, lr=0.0001
loss:  0.0013187596341595054
r2스코어: 0.9999999995313814
rmse:  0.036300950802986134

batch_size=10, lr=0.0005
loss:  0.0011227786308154464
r2스코어: 0.9999999995694826
rmse:  0.033507888453301576

batch_size=10, lr=0.00005
걸린시간:  7.306777000427246
loss:  0.0006204122910276055
r2스코어: 0.9999999997627526
rmse:  0.024874394716944182

RandomSearch
최적의 매개변수:  RandomForestRegressor(min_samples_split=6)
best_params:  {'min_samples_split': 6}
best_score_:  0.99958356686019
model.score:  0.9998318615981776
r2 스코어:  0.9998318615981776
rmse:  20.940408990661812

''' 