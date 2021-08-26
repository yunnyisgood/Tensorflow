import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer


x_data = np.load('./_save/_npy/k55_x_data_diabets.npy')
y_data = np.load('./_save/_npy/k55_y_data_diabets.npy')

print(x_data)
print(y_data)
print(x_data.shape, y_data.shape)
# (442, 10) (442,)


x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,  
train_size=0.8, shuffle=True, random_state=9)

scaler =  QuantileTransformer()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)


# 2. modeling
model = Sequential()
model.add(Dense(512, input_dim=10, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# 3. compile
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=100, batch_size=33, 
validation_split=0.03, shuffle=True)

# 4. evaluate, predict
loss = model.evaluate(x_test, y_test) 
print('loss:', loss)
loss: 2241.13818359375

y_pred = model.predict(x_test)
# print('y예측값: ', y_pred)

r2 = r2_score(y_test, y_pred)
print('r2 스코어: ', r2)
# r2 스코어:  0.5881755653170027

'''
loss: 2125.740966796875
r2 스코어:  0.6093806367072689
'''

