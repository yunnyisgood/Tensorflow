import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
import time
from tensorflow.keras.callbacks import EarlyStopping

# 당뇨 예측

# 1. 데이터
datasets  = load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape) # (442, 10)
print(y.shape) # (442, )

print(datasets.feature_names)
# 'age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
# 10가지의 지수를 통해 당뇨병 지수를 파악한다 


x_train, x_test, y_train, y_test = train_test_split(x, y, 
train_size=0.7, shuffle=True, random_state=9)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_train = scaler.transform(x_train)

# 2. modeling
model = Sequential()
model.add(Dense(64, input_dim=10, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))


model.summary()


# 3. compile

model.compile(loss='mse', optimizer='adam')

model = model.load_weights('./_save/keras46_1_save_weights_new.h5') 
# model.load_weights('./_save/keras46_1_save_weights_2.h5') 
# save_weights는 순수하게 weight 자체만 저장하기 때문에
# compile을 실행한 다음, load_weights를 실행해줘야 한다 

es = EarlyStopping(monitor='val_loss', patience=8, verbose=1, mode='min')

start_time = time.time()
# model.fit(x_train, y_train, epochs=50, validation_split=0.03, shuffle=True, batch_size=256)
end_time = time.time() - start_time



# 4. evaluate, predict
loss = model.evaluate(x_test, y_test) # x_test를 통해 예측한 값, 실제 y_test의 값의 차이를 loss
print('loss:', loss)
# loss: 3002.1416015625

y_pred = model.predict(x_test)
print('y예측값: ', y_pred)

r2 = r2_score(y_test, y_pred)
print('r2 스코어: ', r2)


# './_save/keras46_1_save_weights_1.h5'
# loss: 27093.423828125
# r2 스코어:  -3.902657364981736

# './_save/keras46_1_save_weights_2.h5'
# loss: 26247.33984375
#r2 스코어:  -3.7495556329006376