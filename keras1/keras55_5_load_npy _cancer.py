import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score

x_data = np.load('./_save/_npy/k55_x_data_cancer.npy')
y_data = np.load('./_save/_npy/k55_y_data_cancer.npy')

print(x_data)
print(y_data)
print(x_data.shape, y_data.shape)
# (569, 30) (569,)



x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.7, random_state=9, shuffle=True)


scaler = MinMaxScaler()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)


#modeling
model = Sequential()
model.add(Dense(1000, input_dim=30, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid')) 
# 도출되는 y값을 0과 1로 한정짓는다

# compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# loss=mse가 아닌 binary_crossentropy -> 이진분류 방법 


es = EarlyStopping(monitor='loss', patience=20, verbose=1, mode='min')

hist = model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_split=0.2, callbacks=[es])

# evaluate
loss = model.evaluate(x_test, y_test) # evaluate는 metrics도 반환
print('loss: ', loss[0])
print('accuracy: ', loss[1])

print(y_test[-5:-1])
y_pred = model.predict(x_test[-5:-1])

'''
loss:  0.1676560938358307
accuracy:  0.9181286692619324
'''