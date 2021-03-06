import numpy as np
from tensorflow.keras.layers import concatenate, Concatenate
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input


# 과제 4 -> Concatenate 클래스 사용해서 하기 
# import keras.backend as K
# class Concatenate():  
#     #blablabla   
#     def _merge_function(self, inputs):
#         return K.concatenate(inputs, axis=self.axis)


#1. data

x1 = np.array([range(100), range(301, 401), range(1, 101)])
x2 = np.array([range(101, 201), range(411, 511), range(100, 200)])
x1 = np.transpose(x1)
x2 = np.transpose(x2)

# y1 = np.array([range(1001, 1101)])
# y1 = np.transpose(y1)
y1 = np.array(range(1001, 1101))
y2= np.array(range(1901, 2001))

print(x1.shape, x2.shape, y1.shape, y2.shape)  #(100, 3) (100, 3) (100,)

# 만약 train_size가 없다면?-> 디폴트 값 찾아내기
x1_train, x1_test, x2_train, x2_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x1, x2, y1, y2, train_size=0.7, random_state=9)

print(x1_train.shape, x1_test.shape,
      x2_train.shape, x2_test.shape,
      y1_train.shape, y1_test.shape,
      y2_train.shape, y2_test.shape)


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

# class Concatenate를 통해 output 1개로 도출하기
merge1 = Concatenate(axis=1)([output1, output2]) # 23개의 노드로 합쳐짐
merge2 = Dense(10)(merge1)
merge3 = Dense(5, activation='relu')(merge2)
# last_output = Dense(1)(merge3)

# concat 된 상태에서 다시 분리
output21 = Dense(7, name='output21')(merge3) #  여기서는 분기하고 싶은 지점이 merge3
last_output1 = Dense(1, name='last_output1')(output21)

output22 = Dense(7, name='output22')(merge3)
last_output2 = Dense(1, name='last_output2')(output22)

model = Model(inputs=[input1, input2], outputs=[last_output1, last_output2])

model.summary()

# compile
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

model.fit([x1_train, x2_train], [y1_train, y2_train], epochs=100, batch_size=8, verbose=1)

# 평가 예측
results = model.evaluate([x1_test, x2_test], [y1_test, y2_test])
# x1_test, x2_test를 통해 하나의 y_test를 도출한다

# print(results)
print('loss: ',results[0:]) 
# loss:  [6858.64306640625, 891.2557373046875, 5967.38720703125, 26.094615936279297, 67.51670837402344]


# print('metrics[mae]: ',results[1])
