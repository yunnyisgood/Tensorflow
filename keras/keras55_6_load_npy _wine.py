import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, PowerTransformer, QuantileTransformer
from sklearn.metrics import r2_score

x_data = np.load('./_save/_npy/k55_x_data_wine.npy')
y_data = np.load('./_save/_npy/k55_y_data_wine.npy')

print(x_data)
print(y_data)
print(x_data.shape, y_data.shape)
# (178, 13) (178,)

y_data = to_categorical(y_data)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, 
test_size=0.7, random_state=9, shuffle=True)

scaler = QuantileTransformer()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)

# modeling
model = Sequential()
model.add(Dense(1000, input_shape=(13, ), activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax')) 

# compile
model.compile(loss='categorical_crossentropy', optimizer='adam', 
metrics=['accuracy'])

es = EarlyStopping(monitor='loss', patience=20, mode='min', verbose=1)

hist = model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_split=0.2,
callbacks=[es])

# evaluate

loss = model.evaluate(x_test, y_test)
print('loss: ', loss[0])
print('accuracy: ', loss[1])

y_pred = model.predict(x_test[:5])
# y_pred = model.predict(x_test[-5:-1])
print(y_test[:5])
# print(y_test[-5:-1])
print('--------softmax를 통과한 값 --------')
print(y_pred)
