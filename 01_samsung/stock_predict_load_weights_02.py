import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, QuantileTransformer, StandardScaler
from tensorflow.keras.layers import Dense, LSTM, Input, concatenate, Conv1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model
import time
import numpy as np
import datetime


'''
7/26 삼성전자 시가 예측 
'''

# data 
# samsung, sk 필요 데이터 추출
samsung = pd.read_csv('../_data/samsung_20210721.csv', sep=',',index_col='일자', 
header=0,engine='python', encoding='cp949')
samsung = samsung[['시가','고가','저가','종가', '거래량']]
samsung = samsung.sort_values(by='일자', ascending=True)
samsung = samsung.query('"2011/01/03"<= 일자 <= "2021/07/21"')
samsung_y = samsung.query('"2011/01/10"<= 일자 <= "2021/07/21"')

sk = pd.read_csv('../_data/SK_20210721.csv', sep=',', index_col='일자', header=0,
engine='python', encoding='CP949')
sk = sk[['시가','고가','저가','종가','거래량']]
sk = sk.sort_values(by='일자', ascending=True)
sk = sk.query('"2011/01/03"<= 일자 <= "2021/07/21"')

# 판다스 -> 넘파이로 변환 
samsung = samsung.to_numpy()
samsung_y = samsung_y.to_numpy()
sk = sk.to_numpy()


def split_x(dataset, size):
    aaa =[]
    for i in range(len(dataset)-size+1): # range(10-4=6) -> 6번동안 반복. 10개의 데이터를 5개씩 분리하기 위한 방법 
        subset = dataset[i : (i+size)] # dataset[0:5] -> dataset 0부터 4번째 값까지 
        aaa.append(subset)
    return np.array(aaa)

size = 5

samsung = split_x(samsung, size) # (2597, 5, 5)
samsung_y = split_x(samsung_y, size)
print(samsung.shape)

samsung = samsung.reshape(2597*5, 5)
samsung_y = samsung_y.reshape(2592*5, 5)
sk = split_x(sk, size)
sk = sk.reshape(2597*5, 5)


x1 = samsung[:10000]
y = samsung_y[:10000, 0] # 시가 
y= y.flatten() # (10000,) 
x2 = sk[:10000]

x1_pred = samsung[-5:]
x2_pred = sk[-5:]

print(x1.shape, x2.shape, y.shape )  # (10000, 5) (10000, 5) (10000,)

print(x1_pred.shape, x2_pred.shape) # (10, 5) (10, 5)

x1_train, x1_test,x2_train, x2_test, y_train, y_test = train_test_split(x1, x2,  y, 
                                                        train_size=0.8, random_state=9)

# scaling
scaler = MinMaxScaler()
scaler.fit_transform(x1_train)
scaler.fit_transform(x2_train)
scaler.transform(x1_test)
scaler.transform(x2_test)
scaler.transform(x1_pred)
scaler.transform(x2_pred)


# x와 y를 분리 
x1_train = x1_train.reshape(x1_train.shape[0], x1_train.shape[1], 1)
x1_test = x1_test.reshape(x1_test.shape[0], x1_test.shape[1], 1)
x2_train = x2_train.reshape(x2_train.shape[0], x2_train.shape[1], 1)
x2_test = x2_test.reshape(x2_test.shape[0], x2_test.shape[1], 1)


print(x1_train.shape, x1_test.shape,
      x2_train.shape, x2_test.shape,
      y_train.shape, y_test.shape)

# (4000, 5, 1) (1000, 5, 1) (4000, 5, 1) (1000, 5, 1) (4000,) (1000,)

#2. modeling
# 2-1 model 1
input1 = Input(shape=(5, 1))
xx = LSTM(units=100, activation='relu', return_sequences=True)(input1)
xx = Conv1D(32,2, activation='relu')(xx)
xx = Conv1D(32,2, activation='relu')(xx)
xx = LSTM(units=16, activation='relu', return_sequences=True)(xx)
xx = Conv1D(16,2, activation='relu')(xx)
xx = Flatten()(xx)
xx = Dense(100, activation='relu')(xx)
output1 = Dense(16, name='output1', activation='relu')(xx)

# 2-2 model2
input2 = Input(shape=(5, 1))
xx = LSTM(units=100, activation='relu', return_sequences=True)(input1)
xx = Conv1D(32,2, activation='relu')(xx)
xx = Conv1D(32,2, activation='relu')(xx)
xx = LSTM(units=16, activation='relu', return_sequences=True)(xx)
xx = Conv1D(16,2, activation='relu')(xx)
xx = Flatten()(xx)
xx = Dense(100, activation='relu')(xx)
output2 = Dense(16, name='output2', activation='relu')(xx)

merge1 = concatenate([output1, output2]) 
merge2 = Dense(100, activation='relu')(merge1)
merge3 = Dense(16, activation='relu')(merge2)
last_output = Dense(1)(merge3)


model = Model(inputs=[input1, input2], outputs=last_output)

model.summary()

# compile
model.compile(loss='mse', optimizer='adam', metrics=['acc'])

model.load_weights('./_save/ModelCheckPoint2/stock_weight_save0723_1551.h5')

es = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto', restore_best_weights=True)

start_time = time.time()
# model.fit([x1_train, x2_train], y_train, epochs=100, batch_size=512, verbose=1, callbacks=[es, mcp], validation_split=0.2)
end_time = time.time() -start_time

# model.save_weights('./_save/ModelCheckPoint/stock_weight_save2.h5')

# evaluate
results = model.evaluate([x1_test, x2_test], y_test)
print("걸린시간: ", end_time)
print('loss: ',results[0])
print('acc: ',results[1])

y_pred = model.predict([x1_pred, x2_pred])
print('y_pred: ', y_pred)
print('2일 뒤 예측 주가: ', y_pred[-2])



'''
걸린시간:  9.231413841247559
loss:  1527307.25
acc:  0.0
5일 뒤 예측 주가:  [76769.04]

stock_weight_save0723_1524
걸린시간:  13.27272367477417
loss:  1266166.5
acc:  0.0
5일 뒤 예측 주가:  [78294.92]

stock_weight_save0723_1551
걸린시간:  33.63377523422241
loss:  1258995.625
acc:  0.0
5일 뒤 예측 주가:  [79069.34]

'''

