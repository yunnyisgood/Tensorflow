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
import time 
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# 다중분류, One-Hot-Encoding 

datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target

print(x.shape)  # (150, 4)
print(y.shape)  # (150,)
print(y) # -> cancer와 마찬가지로 0, 1로 구성되어있다

''' 
One-Hot-Encoding (150, ) -> (150, 3, 1)
0 -> [1, 0, 0]
1 -> [0, 1, 0]
2 -> [0, 0, 1]

ex) [0, 1, 2, 1]
=> [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]]
=> (4, )에서 (4, 3)으로 바뀌었다

-> 원래 y 데이터 (150, )에서 (150, 3)으로 바뀌게 됨을 알 수 있다
'''
y = to_categorical(y) # -> one-Hot-Encoding 하는 방법
print(y[:5])

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=9, shuffle=True)


scaler = MinMaxScaler()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)


#modeling
model = Sequential()
model.add(Dense(1000, input_shape=(4,), activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax')) 
# output 

# 3. compile
optimizer = Adam(lr=0.001)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto', verbose=1, factor=0.5)

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=1000, verbose=1, callbacks=[es, reduce_lr], validation_split=0.2,
shuffle=True, batch_size=64)
end_time = time.time() - start_time

# evaluate
loss = model.evaluate(x_test, y_test) # evaluate는 metrics도 반환
print('loss: ', loss[0])
print('accuracy: ', loss[1])

y_pred = model.predict(x_test[:5])
print(y_test[:5])
print('--------softmax를 통과한 값 --------')
print(y_pred)

'''
loss:  0.10622021555900574
accuracy:  0.9555555582046509

loss:  0.03192999213933945
accuracy:  1.0

'''