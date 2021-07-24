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
7/23 삼성전자 종가 예측 
'''


# data 
# samsung, sk 필요 데이터 추출
samsung = pd.read_csv('../_data/samsung_20210721.csv', sep=',',index_col='일자', 
header=0,engine='python', encoding='cp949')
samsung = samsung[['시가','고가','저가','종가', '거래량']]
samsung = samsung.sort_values(by='일자', ascending=True)
samsung = samsung.query('"2011/01/03"<= 일자 <= "2021/07/21"')
print(samsung)

sk = pd.read_csv('../_data/SK_20210721.csv', sep=',', index_col='일자', header=0,
engine='python', encoding='CP949')
sk = sk[['시가','고가','저가','종가','거래량']]
sk = sk.sort_values(by='일자', ascending=True)
sk = sk.query('"2011/01/03"<= 일자 <= "2021/07/21"')
print(sk)

# 판다스 -> 넘파이로 변환 
samsung = samsung.to_numpy()
sk = sk.to_numpy()


def split_x(dataset, size):
    aaa =[]
    for i in range(len(dataset)-size+1): # range(10-4=6) -> 6번동안 반복. 10개의 데이터를 5개씩 분리하기 위한 방법 
        subset = dataset[i : (i+size)] # dataset[0:5] -> dataset 0부터 4번째 값까지 
        aaa.append(subset)
    return np.array(aaa)

size = 5

samsung = split_x(samsung, size) # (2597, 5, 5)
samsung = samsung.reshape(2597*5, 5) 
sk = split_x(sk, size)
sk = sk.reshape(2597*5, 5)


x1 = samsung[:10000]
y = samsung[2:10002, 3] # 종가 
y= y.flatten() # (10000,) 
print(x1.shape, y.shape) # (10000, 5) (10000,)
print(y) # [18840. 18600. 18420. ... 47150. 45950. 46900.]

x2 = sk[:10000]

x1_pred = samsung[-5:]
x2_pred = sk[-5:]

print(x1.shape, x2.shape, y.shape )  # (10000, 5) (10000, 5) (10000,)

print(x1_pred.shape, x2_pred.shape) # (5, 5) (5, 5)

x1_train, x1_test,x2_train, x2_test, y_train, y_test = train_test_split(x1, x2,  y, 
                                                        train_size=0.7, random_state=9)

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

# (7000, 5, 1) (3000, 5, 1) (7000, 5, 1) (3000, 5, 1) (7000,) (3000,)

#2. modeling
# 2-1 model 1
input1 = Input(shape=(5, 1))
xx = LSTM(units=32, activation='relu', return_sequences=True)(input1)
xx = Conv1D(16, 2, activation='relu')(xx)
xx = Conv1D(16,2, activation='relu')(xx)
xx = LSTM(units=16, activation='relu', return_sequences=True)(input1)
xx = Conv1D(8, 2, activation='relu')(xx)
xx = Flatten()(xx)
xx = Dense(100, activation='relu')(xx)
output1 = Dense(16, name='output1', activation='relu')(xx)

# 2-2 model2
input2 = Input(shape=(5, 1))
xx = LSTM(units=32, activation='relu', return_sequences=True)(input1)
xx = Conv1D(16,2, activation='relu')(xx)
xx = Conv1D(16,2, activation='relu')(xx)
xx = LSTM(units=16, activation='relu', return_sequences=True)(input1)
xx = Conv1D(8, 2, activation='relu')(xx)
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

es = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto', restore_best_weights=True)

date = datetime.datetime.now()
date_time = date.strftime("%m%d_%H%M")

filepath = './_save/ModelCheckPoint/' 
filename = '.{epoch:04d}-{val_loss:.4f}.hdf5'           
MCPpath = "".join([filepath, "stock_", date_time, "-", filename])    
#                               47번째 파일 , 오늘 날짜, 

mcp = ModelCheckpoint(monitor='val_loss', save_best_only=True, verbose=1, mode='auto',
                        filepath = MCPpath)

start_time = time.time()
model.fit([x1_train, x2_train], y_train, epochs=100, batch_size=512, verbose=1, callbacks=[es, mcp],
validation_split=0.2)
end_time = time.time() -start_time

weight_path = './_save/ModelCheckPoint/' 
weight_path = "".join([filepath, "stock_weight_save", date_time, ".h5"])   

model.save_weights(weight_path)
# model.save_weights('./_save/ModelCheckPoint/stock_weight_save3.h5')

# evaluate
results = model.evaluate([x1_test, x2_test], y_test)
print("걸린시간: ", end_time)
print('loss: ',results[0])
print('acc: ',results[1])

y_pred = model.predict([x1_pred, x2_pred])
print('y_pred: ', y_pred)
print('2일 뒤 예측 주가: ', y_pred[-2])


'''
걸린시간:  158.40348315238953
loss:  36625975345152.0
metrics[mae]:  0.0
2일 뒤 예측 주가:  [[10649922.]
 [10986487.]]

loss:  233541447188480.0
acc:  0.0020000000949949026
2일 뒤 예측 주가:  [[0.01100441]

걸린시간:  32.81836247444153
loss:  100830749392896.0
acc:  0.0
2일 뒤 예측 주가: [15897088.  ]]

걸린시간:  30.284231901168823
loss:  213269680.0
acc:  0.0
2일 뒤 예측 주가:  [[78932.74]

걸린시간:  23.702643394470215
loss:  213190944.0
acc:  0.0
2일 뒤 예측 주가:  [[78678.22]


걸린시간:  14.718028545379639
loss:  596690.4375
acc:  0.0
y_pred:  [[80030.516]
 [79512.64 ]
 [79175.19 ]
 [78934.73 ]
 [79192.22 ]]
2일 뒤 예측 주가:  [78934.73]

걸린시간:  14.59900689125061
loss:  594541.375
acc:  0.0
y_pred:  [[79602.234]
 [80140.24 ]
 [78742.26 ]
 [78541.38 ]
 [78810.03 ]]
2일 뒤 예측 주가:  [78541.38]

걸린시간:  15.531609058380127
loss:  583461.1875
acc:  0.0
y_pred:  [[79996.55]
 [79960.65]
 [79009.68]
 [78697.75]
 [78884.07]]
2일 뒤 예측 주가:  [78697.75]



'''

