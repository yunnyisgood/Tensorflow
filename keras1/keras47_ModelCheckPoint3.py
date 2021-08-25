import numpy as np
from tensorflow.keras.layers import concatenate, Concatenate
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from keras.callbacks import ModelCheckpoint, EarlyStopping
import time
from sklearn.metrics import r2_score
from tensorflow.python.keras.saving.save import load_model
import datetime

# ensemble modeling -> 여러개의 모델을 통해 하나의 예측치를 도출하는 것 

#1. data

x1 = np.array([range(100), range(301, 401), range(1, 101)])
x2 = np.array([range(101, 201), range(411, 511), range(100, 200)])
x1 = np.transpose(x1)
x2 = np.transpose(x2)

# y1 = np.array([range(1001, 1101)])
# y1 = np.transpose(y1)
y = np.array(range(1001, 1101))

# print(x1.shape, x2.shape, y.shape)  #(100, 3) (100, 3) (100,)

# 만약 train_size가 없다면?-> 디폴트 값 찾아내기
x1_train, x1_test,x2_train, x2_test, y_train, y_test = train_test_split(x1, x2,  y, 
                                                        train_size=0.7, random_state=9)

# print(x1_train.shape, x1_test.shape, x2_train.shape, x2_test.shape, y_train.shape, y_test.shape)

#2. modeling
# 2-1 model 1
input1 = Input(shape=(3, ))
dense1 = Dense(5, activation='relu', name='dense1')(input1)
dense2 = Dense(3, activation='relu', name='dense2')(dense1)
dense3 = Dense(2, activation='relu', name='dense3')(dense2)
output1 = Dense(3, name='output1')(dense3)

# 2-2 model2
input2 = Input(shape=(3, ))
dense11 = Dense(4, activation='relu', name='dense11')(input2)
dense12 = Dense(4, activation='relu', name='dense12')(dense11)
dense13 = Dense(4, activation='relu', name='dense13')(dense12)
dense14 = Dense(4, activation='relu', name='dense14')(dense13)
output2 = Dense(4, name='output2')(dense14)

merge1 = concatenate([output1, output2]) # 23개의 노드로 합쳐짐
merge2 = Dense(10)(merge1)
merge3 = Dense(5, activation='relu')(merge2)
last_output = Dense(1)(merge3)

model = Model(inputs=[input1, input2], outputs=last_output)

model.summary()

# compile
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

es = EarlyStopping(monitor='val_loss', mode='auto', patience=10, verbose=1,
                        restore_best_weights=True)

date = datetime.datetime.now()
date_time = date.strftime("%m%d_%H%M")

filepath = './_save/ModelCheckPoint/' 
filename = '.{epoch:04d}-{val_loss:.4f}.hdf5'           
modelpath = "".join([filepath, "k47_", date_time, "-", filename])    
#                               47번째 파일 , 오늘 날짜, 

mcp = ModelCheckpoint(monitor='val_loss', save_best_only=True, verbose=1, mode='auto',
                        filepath = modelpath)

model.fit([x1_train, x2_train], y_train, epochs=100, batch_size=8, verbose=1, 
                callbacks=[es,mcp], validation_split=0.2)

model.save('./_save/ModelCheckPoint/keras49_model_save2.h5')

# # 평가 예측
print("================ 1. 기본 출력  ================")
results = model.evaluate([x1_test, x2_test], y_test)

print('loss: ',results[0])

y_pred = model.predict([x1_test, x2_test])

r2 = r2_score(y_test, y_pred)
print('r2 스코어: ', r2)

'''
<restore_best_weights=False> 

loss:  5755.94384765625
r2 스코어:  -8.22334925295475

<restore_best_weights=True> 

loss:  3461.4677734375
r2 스코어:  -4.546670677164101
'''

print("================ 2. load_model  ================")

model2 = load_model('./_save/ModelCheckPoint/keras49_model_save2.h5')

results = model2.evaluate([x1_test, x2_test], y_test)

print('loss: ',results[0])

y_pred = model2.predict([x1_test, x2_test])

r2 = r2_score(y_test, y_pred)
print('r2 스코어: ', r2)

'''
<restore_best_weights=False> 

loss:  5755.94384765625
r2 스코어:  -8.22334925295475

<restore_best_weights=True> 

loss:  3461.4677734375
r2 스코어:  -4.546670677164101
'''

print("================ 3. Model CheckPoint  ================")

model3 = load_model('./_save/ModelCheckPoint/keras49_MCP2.h5')

results = model3.evaluate([x1_test, x2_test], y_test)

print('loss: ',results[0])

y_pred = model3.predict([x1_test, x2_test])

r2 = r2_score(y_test, y_pred)
print('r2 스코어: ', r2)

'''
<restore_best_weights=False> 

loss:  6453.1552734375
r2 스코어:  -9.340563439411516

<restore_best_weights=True> 

loss:  3461.4677734375
r2 스코어:  -4.546670677164101
'''