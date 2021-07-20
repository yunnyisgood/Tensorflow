import numpy as np
from numpy import array
from tensorflow.keras.layers import concatenate, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, GRU

# data
x1 = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
             [5,6,7], [6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],
             [20,30,40], [30,40,50], [40,50,60]]) 
x2 = np.array([[10, 20, 30], [20,30,40], [30,40,50], [40,50,60],
             [50,60,70], [60,70,80],[70,80,90],[80,90,100],[90,100,110],[100,110,120],
             [2,3,4], [3,4,5], [4,5,6]]) 
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x1_pred = np.array([55,65,75])
x2_pred = np.array([65,75,85])

print(x1.shape, x2.shape, y.shape) # (13, 3) (13, 3) (13,)

x1 = x1.reshape(x1.shape[0], x1.shape[1], 1)
x2 = x2.reshape(x2.shape[0], x2.shape[1], 1)
x1_pred = x1_pred.reshape(1, x1_pred.shape[0], 1)
x2_pred = x2_pred.reshape(1, x2_pred.shape[0], 1)

print(x1.shape, x2.shape, x1_pred.shape) # (13, 3, 1) (13, 3, 1) (1, 3, 1)

#2. modeling
# 2-1 model 1
input1 = Input(shape=(3, 1))
dense = GRU(units=32, activation='relu', name='dense')(input)
dense1 = Dense(16, name='dense1')(dense) 
dense2 = Dense(16, name='dense2')(dense1)
dense3 = Dense(16, name='dense3')(dense2)
dense4 = Dense(8, name='dense4')(dense3)
output1 = Dense(8, name='output1')(dense4)

# 2-2 model2
input2 = Input(shape=(3, 1))
dense = GRU(units=32, activation='relu', name='dense')(input)
dense11 = Dense(16, name='dense11')(dense) 
dense22 = Dense(16, name='dense22')(dense11)
dense33 = Dense(16, name='dense33')(dense22)
dense44 = Dense(8, name='dense44')(dense33)
output2 = Dense(8, name='output2')(dense44)

# concatenate를 통해 output 1개로 도출하기
merge1 = concatenate([output1, output2]) # 23개의 노드로 합쳐짐
merge2 = Dense(10)(merge1)
merge3 = Dense(5, activation='relu')(merge2)
# last_output = Dense(1)(merge3)

# concat 된 상태에서 다시 분리
output21 = Dense(7)(merge3) #  여기서는 분기하고 싶은 지점이 merge3
last_output1 = Dense(1)(output21)

output22 = Dense(7)(merge3)
last_output2 = Dense(1)(output22)

model = Model(inputs=[input1, input2], outputs=[last_output1, last_output2])

model.summary()

# compile
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

model.fit([x1,x2], y, epochs=100, batch_size=8, verbose=1)

# # 평가 예측
results = model.evaluate([x1, x2], y)
# x1_test, x2_test를 통해 하나의 y_test를 도출한다

# print(results)
print('loss: ',results[0])
print('metrics[mae]: ',results[1])