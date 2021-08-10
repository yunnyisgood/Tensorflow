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
from sklearn.svm import LinearSVC
from sklearn.metrics import r2_score, accuracy_score

# 다중분류, One-Hot-Encoding 

datasets = load_iris()


x = datasets.data
y = datasets.target

# y = to_categorical(y) # -> one-Hot-Encoding 하는 방법
# print(y[:5])
# 머신러닝에서는 y값들을 기본적으로 1차원으로 받아들이기 때문에
# one hot encoding을 하지 않는다 



x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=9, shuffle=True)


scaler = MinMaxScaler()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)


#modeling
model = LinearSVC()

# compile
model.fit(x_train, y_train)

result = model.score(x_test, y_test)
print(result)
# 즉 score는 accuracy를 도출한다 

# loss = model.evaluate(x_test, y_test) # evaluate는 metrics도 반환
# print('loss: ', loss[0])
# print('accuracy: ', loss[1])

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print('accuracy_score: ', acc)

y_pred2 = model.predict(x_test[:5])
print(y_test[:5])
print('--------softmax를 통과한 값 --------')
print(y_pred2)



'''
loss:  0.10622021555900574
accuracy:  0.9555555582046509


C:\ProgramData\Anaconda3\lib\site-packages\sklearn\svm\_base.py:985: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn("Liblinear failed to converge, increase "
1.0
accuracy_score:  1.0
[2 1 2 2 1]
--------softmax를 통과한 값 --------
[2 1 2 2 1]

'''