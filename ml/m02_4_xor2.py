from sklearn.svm import LinearSVC, SVC
import numpy as np
from sklearn.metrics import accuracy_score

#1.data -> or gate [1이 하나라도 있으면 1이 된다 ]
x_data = [[0,0], [0,1], [1,0], [1,1]]
y_data = [0,1,1,0]

#2.modeling
model = SVC()

#3. training
model.fit(x_data, y_data)

#4.predict
y_pred = model.predict(x_data)
print(x_data,'의 예측결과 : ', y_pred)

result = model.score(x_data, y_data)
print('model.score: ', result)

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
'''

