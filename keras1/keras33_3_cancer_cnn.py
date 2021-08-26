import numpy as np
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Conv2D, Dropout, GlobalAveragePooling2D
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
from tensorflow.keras.callbacks import EarlyStopping
import time

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

print(x_train.shape, x_test.shape) # (398, 30) (171, 30)
print(y_train.shape, y_test.shape)

x_train = x_train.reshape(398, 30, 1, 1)
x_test = x_test.reshape(171, 30, 1, 1)



# modeling
model = Sequential()
model.add(Conv2D(filters=6, kernel_size=(1,1), 
                    padding='same', input_shape=(1, 1, 1), activation='relu'))
# model.add(Dropout(0, 2)) # 20%의 드롭아웃의 효과를 낸다 
model.add(Dropout(0.2))
model.add(Conv2D(16, (1,1), padding='same', activation='relu'))   

model.add(Conv2D(64, (1,1),padding='valid', activation='relu'))  
model.add(Dropout(0.2))
model.add(Conv2D(64, (1,1), padding='same', activation='relu')) 


model.add(Conv2D(128, (1,1), padding='valid', activation='relu')) 
model.add(Dropout(0.2))
model.add(Conv2D(128, (1,1), padding='same', activation='relu')) 

# 여기까지가 convolutional layer 

model.add(GlobalAveragePooling2D())
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


'''
loss:  0.19166618585586548
accuracy:  0.9298245906829834

CNN으로 실행했을 때 
loss:  0.401380717754364
accuracy:  0.847953200340271
'''



