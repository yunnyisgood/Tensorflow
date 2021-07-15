import numpy as np
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

datasets = load_breast_cancer() # (569, 30)

print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target

print(x.shape) # (569, 30)
print(y.shape) # (569,)

print(np.unique(y)) # y데이터는 0과 1데이터로만 구성되어 있다


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=9, shuffle=True)


scaler = MinMaxScaler()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)


#modeling
model = Sequential()
model.add(Dense(1000, input_dim=30, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid')) 
# 도출되는 y값을 0과 1로 한정짓는다

# compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# loss=mse가 아닌 binary_crossentropy -> 이진분류 방법 


es = EarlyStopping(monitor='loss', patience=20, verbose=1, mode='min')

hist = model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_split=0.2, callbacks=[es])

# evaluate
loss = model.evaluate(x_test, y_test) # evaluate는 metrics도 반환
print('loss: ', loss[0])
print('accuracy: ', loss[1])

print(y_test[-5:-1])
y_pred = model.predict(x_test[-5:-1])
print(y_pred)

# loss:  0.19166618585586548
# accuracy:  0.9298245906829834

# y_pred = model.predict(x_test)
# r2 = r2_score(y_test, y_pred)
# print('r2 스코어: ', r2)

# plt.plot(hist.history['loss'])
# plt.plot(hist.history['val_loss'])

# plt.title("loss, val_loss")
# plt.xlabel("epochs")
# plt.ylabel("loss, val_loss")
# plt.legend(["train loss", "val_loss"])
# plt.show()

