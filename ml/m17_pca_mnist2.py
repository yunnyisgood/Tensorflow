import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Reshape, Conv1D
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, QuantileTransformer, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping

'''
실습
pca를 통해 0.95 이상인 n_components가 몇개?
'''

(x_train, y_train), (x_test, y_test) = mnist.load_data() 
# 

print(x_train.shape, x_test.shape)
# (60000, 28, 28) (10000, 28, 28)

x = np.append(x_train, x_test, axis=0)
print(x.shape) # (70000, 28, 28)
print(type(x))  # <class 'numpy.ndarray'>    

x = x.reshape(70000, 28*28)

'''pca = PCA(n_components=154)
x = pca.fit_transform(x)
print(x)
print(x.shape)

pca_EVR = pca.explained_variance_ratio_
print(pca_EVR)
# [0.40242142 0.14923182 0.12059623 0.09554764 0.06621856 0.06027192
#  0.05365605 0.04336832]

print(sum(pca_EVR))
# 0.9661701241516084

cumsum = np.cumsum(pca_EVR)
print(cumsum)
# [0.40242142 0.55165324 0.67224947 0.76779711 0.83401567 0.89428759
#  0.94794364 0.99131196]

print(np.argmax(cumsum >= 0.95)+1)'''
# 154

x_train, x_test, y_train, y_test = train_test_split(
    x, np.concatenate((y_train, y_test)), train_size=0.8, random_state=66, shuffle=True
)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (56000, 154) (14000, 154) (56000,) (14000,)

# modeling
model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(784, )))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(10, activation='softmax'))

# compile   
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')

model.fit(x_train , y_train, epochs=1000, verbose=1, callbacks=[es], validation_split=0.01,
shuffle=True, batch_size=256)

# evaluate -> predict 할 필요는 없다
loss = model.evaluate(x_test, y_test)
print('loss: ', loss[0])
print('accuracy: ', loss[1])

'''
압축 전:
loss:  2.3022656440734863
accuracy:  0.10971428453922272

압축 후 DNN:
loss:  0.34136977791786194
accuracy:  0.946142852306366
'''