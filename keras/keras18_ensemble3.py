import numpy as np
from tensorflow.keras.layers import concatenate, Concatenate
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

# ensemble modeling -> 여러개의 모델을 통해 하나의 예측치를 도출하는 것 

#1. data

x1 = np.array([range(100), range(301, 401), range(1, 101)])
# x2 = np.array([range(101, 201), range(411, 511), range(100, 200)])
x1 = np.transpose(x1)
# x2 = np.transpose(x2)

# y1 = np.array([range(1001, 1101)])
# y1 = np.transpose(y1)
y1 = np.array(range(1001, 1101))
y2= np.array(range(1901, 2001))

print(x1.shape, y1.shape, y2.shape)  #(100, 3) (100, 3) (100,)

# 만약 train_size가 없다면?-> 디폴트 값 찾아내기
x1_train, x1_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x1, y1, y2, train_size=0.7, random_state=9)

print(x1_train.shape, x1_test.shape,
      y1_train.shape, y1_test.shape,
      y2_train.shape, y2_test.shape)


#2. modeling
# 2-1 model 1
input = Input(shape=(3, ))
dense1 = Dense(100, activation='relu', name='dense1')(input)
dense2 = Dense(50, activation='relu', name='dense2')(dense1)
dense3 = Dense(20, activation='relu', name='dense3')(dense2)
output1 = Dense(15, name='output1')(dense3)


# concat 된 상태에서 다시 분리
output2 = Dense(7, name='output2')(output1) #  여기서는 분기하고 싶은 지점이 merge3
last_output1 = Dense(1, name='last_output1')(output2)

output3 = Dense(7, name='output3')(output1)
last_output2 = Dense(1, name='last_output2')(output3)

model = Model(inputs=input, outputs=[last_output1, last_output2])

model.summary()

# compile
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

model.fit(x1_train, [y1_train, y2_train], epochs=100, batch_size=1, verbose=1)

# # 평가 예측
results = model.evaluate(x1_test, [y1_test, y2_test])
# x1_test, x2_test를 통해 하나의 y_test를 도출한다

print('loss: ',results)
# batch_size=8일 떄
# [3.0434792041778564, 2.5818612575531006, 0.46161791682243347, 1.3974140882492065, 0.587158203125] 
# batch_size=1일 떄
# [0.0006406679749488831, 0.0006294503691606224, 1.1217593964829575e-05, 0.02387288399040699, 0.0031575520988553762]
# print('metrics[mae]: ',results[1])
