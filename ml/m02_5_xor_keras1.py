from sklearn.svm import LinearSVC, SVC
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1.data -> or gate [1이 하나라도 있으면 1이 된다 ]
x_data = [[0,0], [0,1], [1,0], [1,1]]
y_data = [0,1,1,0]

#2.modeling
# model = SVC()
model = Sequential()
model.add(Dense(1, input_dim = 2, activation='sigmoid'))

#3. training
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, batch_size=1, epochs=100)

#4.predict
y_pred = model.predict(x_data)
print(x_data,'의 예측결과 : ', y_pred)

result = model.evaluate(x_data, y_data)
print('loss: ', result[1])

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

Traceback (most recent call last):
  File "d:\Tensorflow\ml\m02_5_xor_keras.py", line 27, in <module>
    acc = accuracy_score(y_data, y_pred)
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\validation.py", line 63, in inner_f
    return f(*args, **kwargs)
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\metrics\_classification.py", line 202, in accuracy_score
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
  File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\metrics\_classification.py", line 92, in _check_targets
    raise ValueError("Classification metrics can't handle a mix of {0} "
ValueError: Classification metrics can't handle a mix of binary and continuous targets 

=> 다음 파일에서 해결 
'''

