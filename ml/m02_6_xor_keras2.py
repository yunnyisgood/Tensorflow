from sklearn.svm import LinearSVC, SVC
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

#1.data -> or gate [1이 하나라도 있으면 1이 된다 ]
x_data = [[0,0], [0,1], [1,0], [1,1]]
y_data = [0,1,1,0]

#2.modeling -> 다층 레이어 구성해서 acc =1이 나오도록 구성!
# model = SVC()
model = Sequential()
model.add(Dense(100, input_dim = 2, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid')) 

#3. training
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, batch_size=1, epochs=100)

#4.predict
y_pred = model.predict(x_data)
print(x_data,'의 예측결과 1 : ', y_pred)

y_pred = np.rint(y_pred) # ->  반올림을 통해 0 아니면 1의 값으로 y_pred를 변형 
print(x_data,'의 예측결과 : ', y_pred)

result = model.evaluate(x_data, y_data)
print('acc: ', result[1])

acc = accuracy_score(y_data, y_pred)
print('accuracy_score: ', acc)

'''
[[0, 0], [0, 1], [1, 0], [1, 1]] 의 예측결과 :  [0 1 1 1]
model.score:  0.75
accuracy_score:  0.75

LIinear SVC -> SVC 다층에 적용되는 모델로 변경
[[0, 0], [0, 1], [1, 0], [1, 1]] 의 예측결과 :  [0 1 1 0]
model.score:  1.0
accuracy_score:  1.0

y_pred를 0, 1로 반올림했을 때 accuracy_score 사용 가능!
[[0, 0], [0, 1], [1, 0], [1, 1]] 의 예측결과 1 :  [[0.00557208]
 [0.9986767 ]
 [0.9979791 ]
 [0.00222901]]
[[0, 0], [0, 1], [1, 0], [1, 1]] 의 예측결과 :  [[0.]
 [1.]
 [1.]
 [0.]]
acc:  1.0
accuracy_score:  1.0
'''

