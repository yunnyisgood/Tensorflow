import numpy as np
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error


# data 

x_data = np.array(range(1, 101)) # 연속적 데이터 
x_pred = np.array(range(96, 105))
'''   96, 97, 98, 99, 100    ?   '''
size = 5


def split_x(dataset, size):
    aaa =[]
    for i in range(len(dataset)-size): # 100-5 = 25번동안 반복 
        subset = dataset[i : (i+size+1)] # dataset[0:7] -> 한행에 6개의 값 
        subset = dataset[i : (i+size+1)] # dataset[0:7] -> 한행에 6개의 값 
        aaa.append(subset)
    return np.array(aaa)

dataset = split_x(x_data, size)

print(dataset)

x = dataset[:, :5] #열은 없고, 처음부터 4번째 전까지의 행을 출력
y = dataset[:, 5]

x_train, x_test, y_train, y_test = train_test_split(x, y, 
train_size=0.8, shuffle=True, random_state=9)

print(x.shape, y.shape) # (95, 5) (95,)

# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)


print(x_train.shape,x_test.shape,y_train.shape,y_test.shape,) # (76, 5) (19, 5) (76,) (19,)


# modeling
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(5, )))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))


# compile 
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=100, batch_size=33, 
validation_split=0.03, shuffle=True)

# evaluate, predict
loss = model.evaluate(x_test, y_test) 
print('loss:', loss)

y_pred = model.predict(x_test)
print('y예측값: ', y_pred)

r2 = r2_score(y_test, y_pred)
print('r2 스코어: ', r2)

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))
rmse = RMSE(y_test, y_pred)
print('rmse: ', rmse)

'''
> LSTM
loss: 0.26612091064453125
r2 스코어:  0.9995302906988577
rmse:  0.5158690684091842

> DNN
loss: 0.0003202072693966329
r2 스코어:  0.9999994348270251
rmse:  0.01789433657098805
'''